bool MetalComputeBackend::dispatch_acoustic_pressure_eos_cartesian_agrid(
    const AcousticPressureEosCartesianAgridDispatch& d)
{
    @autoreleasepool
    {
        std::size_t cells = 0, bytes = 0;
        if (!supports(DynamicsKernelOp::AcousticPressureEosCartesianAgrid) || d.nx < 4 || d.ny < 4 || d.nz < 4 ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || d.rho_in == nullptr || d.theta == nullptr ||
            d.drho_dt == nullptr || d.rho_out == nullptr || d.p_out == nullptr || d.rho_out == d.p_out ||
            !std::isfinite(d.dt_small) || d.dt_small <= 0.0f || !std::isfinite(d.reference_pressure) ||
            d.reference_pressure <= 0.0f || !std::isfinite(d.gas_constant_dry_air) || d.gas_constant_dry_air <= 0.0f ||
            !std::isfinite(d.gamma) || d.gamma <= 0.0f || !(d.rho_min > 0.0f && d.rho_max > d.rho_min) ||
            !(d.p_min > 0.0f && d.p_max > d.p_min) || bytes > impl_->max_buffer_bytes())
            return false;
        for (std::size_t n = 0; n < cells; ++n)
            if (!std::isfinite(d.rho_in[n]) || d.rho_in[n] <= 0.0f || !std::isfinite(d.theta[n]) ||
                d.theta[n] <= 0.0f || !std::isfinite(d.drho_dt[n]))
                return false;

        const float*  source[3] = {d.rho_in, d.theta, d.drho_dt};
        id<MTLBuffer> in[3]     = {};
        for (int i = 0; i < 3; ++i)
        {
            in[i] =
                [impl_->hardware.device newBufferWithBytes:source[i] length:bytes options:MTLResourceStorageModeShared];
            if (in[i] == nil)
                return false;
        }
        id<MTLBuffer> out[2] = {
            [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared]};
        if (out[0] == nil || out[1] == nil)
            return false;

        AcousticAgridPressureEosConstants c{};
        c.nx                            = d.nx;
        c.ny                            = d.ny;
        c.nz                            = d.nz;
        c.dt_small                      = d.dt_small;
        c.reference_pressure            = d.reference_pressure;
        c.gas_constant_dry_air          = d.gas_constant_dry_air;
        c.gamma                         = d.gamma;
        c.rho_min                       = d.rho_min;
        c.rho_max                       = d.rho_max;
        c.p_min                         = d.p_min;
        c.p_max                         = d.p_max;
        id<MTLCommandBuffer>         cb = impl_->command_buffer();
        id<MTLComputeCommandEncoder> e  = cb ? [cb computeCommandEncoder] : nil;
        const NSUInteger             maximum =
            impl_->pipeline(KernelId::AcousticPressureEosCartesianAgrid).maxTotalThreadsPerThreadgroup;
        if (e == nil || maximum == 0)
            return false;
        [e setComputePipelineState:impl_->pipeline(KernelId::AcousticPressureEosCartesianAgrid)];
        for (NSUInteger i = 0; i < 3; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 2; ++i)
            [e setBuffer:out[i] offset:0 atIndex:i + 3];
        [e setBytes:&c length:sizeof(c) atIndex:5];
        [e dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [e endEncoding];
        if (!impl_->finish(cb))
            return false;
        const float* next_rho = static_cast<const float*>(out[0].contents);
        const float* next_p   = static_cast<const float*>(out[1].contents);
        for (std::size_t n = 0; n < cells; ++n)
            if (!std::isfinite(next_rho[n]) || next_rho[n] <= 0.0f || !std::isfinite(next_p[n]) || next_p[n] <= 0.0f)
                return false;
        if (d.completed_observer != nullptr)
            d.completed_observer(d.completed_observer_context, next_rho, next_p);
        std::memcpy(d.rho_out, next_rho, bytes);
        std::memcpy(d.p_out, next_p, bytes);
        ++impl_->telemetry.completed_acoustic_agrid_pressure_eos_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_acoustic_momentum_cartesian_agrid(const AcousticMomentumCartesianAgridDispatch& d)
{
    @autoreleasepool
    {
        std::size_t  cells = 0, bytes = 0;
        const float* source[6] = {d.rho, d.p, d.p0_base_data, d.u_in, d.v_in, d.w_in};
        if (!supports(DynamicsKernelOp::AcousticMomentumCartesianAgrid) || d.nx < 4 || d.ny < 4 || d.nz < 4 ||
            d.base_len < d.nz || !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || d.u_out == nullptr ||
            d.v_out == nullptr || d.w_out == nullptr || d.u_out == d.v_out || d.u_out == d.w_out ||
            d.v_out == d.w_out || !std::isfinite(d.dx) || d.dx <= 0.0f || !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dz) || d.dz <= 0.0f || !std::isfinite(d.dt_small) || d.dt_small <= 0.0f ||
            !std::isfinite(d.wind_clamp_h) || d.wind_clamp_h <= 0.0f || !std::isfinite(d.wind_clamp_v) ||
            d.wind_clamp_v <= 0.0f || bytes > impl_->max_buffer_bytes())
            return false;
        for (const float* value : source)
            if (value == nullptr)
                return false;
        for (std::size_t n = 0; n < cells; ++n)
            if (!std::isfinite(d.rho[n]) || d.rho[n] <= 0.0f || !std::isfinite(d.p[n]) || !std::isfinite(d.u_in[n]) ||
                !std::isfinite(d.v_in[n]) || !std::isfinite(d.w_in[n]))
                return false;
        for (int k = 0; k < d.nz; ++k)
            if (!std::isfinite(d.p0_base_data[k]))
                return false;

        const std::size_t profile_bytes = static_cast<std::size_t>(d.nz) * sizeof(float);
        id<MTLBuffer>     in[6]         = {};
        for (int i = 0; i < 6; ++i)
        {
            in[i] = [impl_->hardware.device newBufferWithBytes:source[i]
                                                        length:i == 2 ? profile_bytes : bytes
                                                       options:MTLResourceStorageModeShared];
            if (in[i] == nil)
                return false;
        }
        id<MTLBuffer> out[3] = {};
        for (int i = 0; i < 3; ++i)
        {
            out[i] = [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (out[i] == nil)
                return false;
        }
        AcousticAgridNativeMomentumConstants c{};
        c.nx                            = d.nx;
        c.ny                            = d.ny;
        c.nz                            = d.nz;
        c.dx                            = d.dx;
        c.dy                            = d.dy;
        c.dz                            = d.dz;
        c.dt_small                      = d.dt_small;
        c.wind_clamp_h                  = d.wind_clamp_h;
        c.wind_clamp_v                  = d.wind_clamp_v;
        id<MTLCommandBuffer>         cb = impl_->command_buffer();
        id<MTLComputeCommandEncoder> e  = cb ? [cb computeCommandEncoder] : nil;
        const NSUInteger             maximum =
            impl_->pipeline(KernelId::AcousticMomentumCartesianAgrid).maxTotalThreadsPerThreadgroup;
        if (e == nil || maximum == 0)
            return false;
        [e setComputePipelineState:impl_->pipeline(KernelId::AcousticMomentumCartesianAgrid)];
        for (NSUInteger i = 0; i < 6; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 3; ++i)
            [e setBuffer:out[i] offset:0 atIndex:i + 6];
        [e setBytes:&c length:sizeof(c) atIndex:9];
        [e dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [e endEncoding];
        if (!impl_->finish(cb))
            return false;
        const float* result[3] = {static_cast<const float*>(out[0].contents),
                                  static_cast<const float*>(out[1].contents),
                                  static_cast<const float*>(out[2].contents)};
        for (int f = 0; f < 3; ++f)
            for (std::size_t n = 0; n < cells; ++n)
                if (!std::isfinite(result[f][n]))
                    return false;
        if (d.completed_observer)
            d.completed_observer(d.completed_observer_context, result[0], result[1], result[2]);
        std::memcpy(d.u_out, result[0], bytes);
        std::memcpy(d.v_out, result[1], bytes);
        std::memcpy(d.w_out, result[2], bytes);
        ++impl_->telemetry.completed_acoustic_agrid_native_momentum_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_acoustic_null_mode_constraint_cartesian_agrid(
    const AcousticNullModeConstraintCartesianAgridDispatch& d)
{
    @autoreleasepool
    {
        std::size_t  cells = 0, bytes = 0;
        const float* scalar[8]   = {d.rho, d.qv, d.qc, d.qr, d.qi, d.qs, d.qg, d.qh};
        const float* velocity[3] = {d.u_in, d.v_in, d.w_in};
        float*       output[3]   = {d.u_out, d.v_out, d.w_out};
        if (!supports(DynamicsKernelOp::AcousticNullModeConstraintCartesianAgrid) || d.nx < 4 || d.ny < 4 || d.nz < 4 ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || bytes > impl_->max_buffer_bytes())
            return false;
        for (const float* value : scalar)
            if (!value)
                return false;
        for (const float* value : velocity)
            if (!value)
                return false;
        for (float* value : output)
            if (!value)
                return false;
        for (std::size_t n = 0; n < cells; ++n)
        {
            if (!std::isfinite(d.rho[n]) || d.rho[n] <= 0.0f)
                return false;
            for (int q = 1; q < 8; ++q)
                if (!std::isfinite(scalar[q][n]) || scalar[q][n] < 0.0f)
                    return false;
            for (int f = 0; f < 3; ++f)
                if (!std::isfinite(velocity[f][n]))
                    return false;
        }
        id<MTLBuffer> buffers[11] = {};
        for (int i = 0; i < 8; ++i)
        {
            buffers[i] =
                [impl_->hardware.device newBufferWithBytes:scalar[i] length:bytes options:MTLResourceStorageModeShared];
            if (!buffers[i])
                return false;
        }
        for (int i = 0; i < 3; ++i)
        {
            buffers[i + 8] = [impl_->hardware.device newBufferWithBytes:velocity[i]
                                                                 length:bytes
                                                                options:MTLResourceStorageModeShared];
            if (!buffers[i + 8])
                return false;
        }
        id<MTLCommandBuffer> cb = impl_->command_buffer();
        if (!cb)
            return false;
        const NSUInteger maximum =
            impl_->pipeline(KernelId::AcousticNullModeConstraintCartesianAgrid).maxTotalThreadsPerThreadgroup;
        if (!maximum)
            return false;
        for (int pass = 0; pass < 3; ++pass)
        {
            AcousticAgridLinePassConstants c{d.nx, d.ny, d.nz, pass};
            std::size_t                    lines = pass == 0   ? std::size_t(d.ny - 2) * (d.nz - 2)
                                                   : pass == 1 ? std::size_t(d.nx - 2) * (d.nz - 2)
                                                               : std::size_t(d.nx - 2) * (d.ny - 2);
            id<MTLComputeCommandEncoder>   e     = [cb computeCommandEncoder];
            if (!e)
                return false;
            [e setComputePipelineState:impl_->pipeline(KernelId::AcousticNullModeConstraintCartesianAgrid)];
            for (NSUInteger i = 0; i < 11; ++i)
                [e setBuffer:buffers[i] offset:0 atIndex:i];
            [e setBytes:&c length:sizeof(c) atIndex:11];
            [e dispatchThreads:MTLSizeMake(lines, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
            [e endEncoding];
        }
        if (!impl_->finish(cb))
            return false;
        const float* result[3] = {static_cast<const float*>(buffers[8].contents),
                                  static_cast<const float*>(buffers[9].contents),
                                  static_cast<const float*>(buffers[10].contents)};
        for (int f = 0; f < 3; ++f)
            for (std::size_t n = 0; n < cells; ++n)
                if (!std::isfinite(result[f][n]))
                    return false;
        if (d.completed_observer)
            d.completed_observer(d.completed_observer_context, result[0], result[1], result[2]);
        for (int f = 0; f < 3; ++f)
            std::memcpy(output[f], result[f], bytes);
        ++impl_->telemetry.completed_acoustic_agrid_constraint_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_acoustic_boundary_cartesian_agrid(const AcousticBoundaryCartesianAgridDispatch& d)
{
    @autoreleasepool
    {
        std::size_t  cells = 0, bytes = 0;
        const float* profile[4] = {d.p0_base_data, d.rho0_base_data, d.u0_base_data, d.v0_base_data};
        float*       state[5]   = {d.u, d.v, d.w, d.rho, d.p};
        if (!supports(DynamicsKernelOp::AcousticBoundaryCartesianAgrid) || d.nx < 4 || d.ny < 4 || d.nz < 4 ||
            d.base_len < d.nz || !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || bytes > impl_->max_buffer_bytes())
            return false;
        for (const float* value : profile)
            if (!value)
                return false;
        for (float* value : state)
            if (!value)
                return false;
        for (int k = 0; k < d.nz; ++k)
            for (const float* value : profile)
                if (!std::isfinite(value[k]))
                    return false;
        for (std::size_t n = 0; n < cells; ++n)
        {
            for (int f = 0; f < 5; ++f)
                if (!std::isfinite(state[f][n]))
                    return false;
            if (state[3][n] <= 0.0f || state[4][n] <= 0.0f)
                return false;
        }
        const std::size_t profile_bytes = std::size_t(d.nz) * sizeof(float);
        id<MTLBuffer>     buffers[9]    = {};
        for (int i = 0; i < 4; ++i)
        {
            buffers[i] = [impl_->hardware.device newBufferWithBytes:profile[i]
                                                             length:profile_bytes
                                                            options:MTLResourceStorageModeShared];
            if (!buffers[i])
                return false;
        }
        for (int i = 0; i < 5; ++i)
        {
            buffers[i + 4] =
                [impl_->hardware.device newBufferWithBytes:state[i] length:bytes options:MTLResourceStorageModeShared];
            if (!buffers[i + 4])
                return false;
        }
        id<MTLCommandBuffer> cb = impl_->command_buffer();
        if (!cb)
            return false;
        const NSUInteger maximum =
            impl_->pipeline(KernelId::AcousticBoundaryCartesianAgrid).maxTotalThreadsPerThreadgroup;
        if (!maximum)
            return false;
        for (int pass = 0; pass < 3; ++pass)
        {
            AcousticAgridLinePassConstants c{d.nx, d.ny, d.nz, pass};
            std::size_t                    lines = pass == 0   ? std::size_t(d.ny) * d.nz
                                                   : pass == 1 ? std::size_t(d.nx) * d.nz
                                                               : std::size_t(d.nx) * d.ny;
            id<MTLComputeCommandEncoder>   e     = [cb computeCommandEncoder];
            if (!e)
                return false;
            [e setComputePipelineState:impl_->pipeline(KernelId::AcousticBoundaryCartesianAgrid)];
            for (NSUInteger i = 0; i < 9; ++i)
                [e setBuffer:buffers[i] offset:0 atIndex:i];
            [e setBytes:&c length:sizeof(c) atIndex:9];
            [e dispatchThreads:MTLSizeMake(lines, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
            [e endEncoding];
        }
        if (!impl_->finish(cb))
            return false;
        const float* result[5] = {static_cast<const float*>(buffers[4].contents),
                                  static_cast<const float*>(buffers[5].contents),
                                  static_cast<const float*>(buffers[6].contents),
                                  static_cast<const float*>(buffers[7].contents),
                                  static_cast<const float*>(buffers[8].contents)};
        for (int f = 0; f < 5; ++f)
            for (std::size_t n = 0; n < cells; ++n)
                if (!std::isfinite(result[f][n]))
                    return false;
        for (std::size_t n = 0; n < cells; ++n)
            if (result[3][n] <= 0.0f || result[4][n] <= 0.0f)
                return false;
        if (d.completed_observer)
            d.completed_observer(d.completed_observer_context, result[0], result[1], result[2], result[3], result[4]);
        for (int f = 0; f < 5; ++f)
            std::memcpy(state[f], result[f], bytes);
        ++impl_->telemetry.completed_acoustic_agrid_boundary_dispatches;
        return true;
    }
}
