// Individual Cartesian C-grid pressure and momentum dispatches.
bool MetalComputeBackend::dispatch_acoustic_pressure_cartesian_cgrid(const AcousticPressureCartesianCgridDispatch& d)
{
    @autoreleasepool
    {
        std::size_t cells = 0, bytes = 0;
        if (!supports(DynamicsKernelOp::AcousticPressureCartesianCgrid) ||
            !valid_acoustic_cgrid_dispatch(d) || !checked_bytes(d.nx, d.ny, d.nz, cells, bytes))
            return false;
        const std::size_t prof    = static_cast<std::size_t>(d.nz) * sizeof(float);
        const void*       data[]  = {d.u, d.v, d.w, d.rho_in, d.p_in, d.rho0_base_data, d.p0_base_data, d.theta_data};
        const std::size_t sizes[] = {bytes, bytes, bytes, bytes, bytes, prof, prof, bytes};
        id<MTLBuffer>     in[8]   = {};
        for (int i = 0; i < 8; ++i)
        {
            in[i] = [impl_->hardware.device newBufferWithBytes:data[i]
                                                        length:sizes[i]
                                                       options:MTLResourceStorageModeShared];
            if (in[i] == nil)
                return false;
        }
        id<MTLBuffer> ro = [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> po = [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer>         cb = impl_->command_buffer();
        id<MTLComputeCommandEncoder> e  = [cb computeCommandEncoder];
        if (ro == nil || po == nil || cb == nil || e == nil ||
            impl_->pipeline(KernelId::AcousticPressureCartesianCgrid).maxTotalThreadsPerThreadgroup == 0)
            return false;
        AcousticPressureCgridConstants c{};
        c.nx                  = d.nx;
        c.ny                  = d.ny;
        c.nz                  = d.nz;
        c.dx                  = d.dx;
        c.dy                  = d.dy;
        c.dz_val              = d.dz;
        c.gamma_val           = static_cast<float>(d.gamma_val);
        c.gamma_low           = static_cast<float>(d.gamma_val - static_cast<double>(c.gamma_val));
        c.dt_small            = d.dt_small;
        c.rho_min             = d.rho_min;
        c.rho_max             = d.rho_max;
        c.p_min               = d.p_min;
        c.p_max               = d.p_max;
        c.p_ref_fallback      = d.p_ref_fallback;
        c.frame_velocity_u_ms = d.frame_velocity_u_ms;
        c.frame_velocity_v_ms = d.frame_velocity_v_ms;
        c.preserve_unrepaired = d.repair_primary_state ? 0 : 1;
        [e setComputePipelineState:impl_->pipeline(KernelId::AcousticPressureCartesianCgrid)];
        for (NSUInteger i = 0; i < 8; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i];
        [e setBuffer:ro offset:0 atIndex:8];
        [e setBuffer:po offset:0 atIndex:9];
        [e setBytes:&c length:sizeof(c) atIndex:10];
        [e dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      std::min<NSUInteger>(64,
                                                           impl_->pipeline(KernelId::AcousticPressureCartesianCgrid)
                                                               .maxTotalThreadsPerThreadgroup),
                                      1,
                                      1)];
        [e endEncoding];
        if (!impl_->finish(cb))
            return false;
        std::memcpy(d.rho_out, ro.contents, bytes);
        std::memcpy(d.p_out, po.contents, bytes);
        impl_->resident.fields.invalidate("rho");
        impl_->resident.fields.invalidate("p");
        ++impl_->telemetry.completed_acoustic_pressure_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_acoustic_momentum_cartesian_cgrid(const AcousticMomentumCartesianCgridDispatch& d)
{
    @autoreleasepool
    {
        std::size_t cells = 0, bytes = 0;
        if (!supports(DynamicsKernelOp::AcousticMomentumCartesianCgrid) ||
            !valid_acoustic_cgrid_dispatch(d) || !checked_bytes(d.nx, d.ny, d.nz, cells, bytes))
            return false;
        std::vector<float> paired_gradient;
        const float* gradient = pack_acoustic_base_gradient(d, paired_gradient);
        const std::size_t prof = static_cast<std::size_t>(d.nz) * sizeof(float) * (d.dp0_dz_low_data ? 2 : 1);
        const void*       data[]  = {d.rho_data, d.p_data, gradient, d.u_in, d.v_in, d.w_in};
        const std::size_t sizes[] = {bytes, bytes, prof, bytes, bytes, bytes};
        id<MTLBuffer>     in[6]   = {};
        for (int i = 0; i < 6; ++i)
        {
            in[i] = [impl_->hardware.device newBufferWithBytes:data[i]
                                                        length:sizes[i]
                                                       options:MTLResourceStorageModeShared];
            if (in[i] == nil)
                return false;
        }
        id<MTLBuffer> out[3] = {};
        for (auto& b : out)
        {
            b = [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (b == nil)
                return false;
        }
        id<MTLCommandBuffer>         cb = impl_->command_buffer();
        id<MTLComputeCommandEncoder> e  = [cb computeCommandEncoder];
        if (cb == nil || e == nil ||
            impl_->pipeline(KernelId::AcousticMomentumCartesianCgrid).maxTotalThreadsPerThreadgroup == 0)
            return false;
        AcousticMomentumCgridConstants c{};
        c.nx           = d.nx;
        c.ny           = d.ny;
        c.nz           = d.nz;
        c.dx           = d.dx;
        c.dy           = d.dy;
        c.dz_val       = d.dz;
        c.dt_small     = d.dt_small;
        c.wind_clamp_h = d.wind_clamp_h;
        c.wind_clamp_v = d.wind_clamp_v;
        c.preserve_unrepaired = d.repair_primary_state ? 0 : 1;
        c.paired_base_gradient = d.dp0_dz_low_data ? 1 : 0;
        [e setComputePipelineState:impl_->pipeline(KernelId::AcousticMomentumCartesianCgrid)];
        for (NSUInteger i = 0; i < 6; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 3; ++i)
            [e setBuffer:out[i] offset:0 atIndex:i + 6];
        [e setBytes:&c length:sizeof(c) atIndex:9];
        [e dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      std::min<NSUInteger>(64,
                                                           impl_->pipeline(KernelId::AcousticMomentumCartesianCgrid)
                                                               .maxTotalThreadsPerThreadgroup),
                                      1,
                                      1)];
        [e endEncoding];
        if (!impl_->finish(cb))
            return false;
        std::memcpy(d.u_out, out[0].contents, bytes);
        std::memcpy(d.v_out, out[1].contents, bytes);
        std::memcpy(d.w_out, out[2].contents, bytes);
        impl_->resident.fields.invalidate("u");
        impl_->resident.fields.invalidate("v");
        impl_->resident.fields.invalidate("w");
        ++impl_->telemetry.completed_acoustic_momentum_dispatches;
        return true;
    }
}
