bool MetalComputeBackend::dispatch_acoustic_continuity_mass_flux_cartesian_agrid(
    const AcousticContinuityMassFluxCartesianAgridDispatch& d)
{
    @autoreleasepool
    {
        std::size_t cells = 0, bytes = 0;
        const float* inputs[] = {d.u, d.v, d.w, d.rho, d.p, d.p0_base_data};
        float* outputs[] = {
            d.drho_dt, d.mass_flux_x, d.mass_flux_y, d.mass_flux_z};
        if (!supports(DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid) ||
            d.nx < 4 || d.ny < 4 || d.nz < 4 || d.base_len < d.nz ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) ||
            !std::isfinite(d.dx) || d.dx <= 0.0f ||
            !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dz) || d.dz <= 0.0f ||
            !std::isfinite(d.dt_small) || d.dt_small <= 0.0f ||
            !std::isfinite(d.mass_flux_weight) ||
            (d.accumulate_mass_flux && d.mass_flux_weight <= 0.0f) ||
            !std::isfinite(d.frame_velocity_u_ms) ||
            !std::isfinite(d.frame_velocity_v_ms) ||
            d.lateral_bc_west < 0 || d.lateral_bc_west > 5 ||
            d.lateral_bc_east < 0 || d.lateral_bc_east > 5 ||
            d.lateral_bc_south < 0 || d.lateral_bc_south > 5 ||
            d.lateral_bc_north < 0 || d.lateral_bc_north > 5 ||
            d.top_bc < 0 || d.top_bc > 2 || bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        for (const float* input : inputs)
            if (input == nullptr) return false;
        for (int out = 0; out < 4; ++out)
        {
            if (outputs[out] == nullptr) return false;
            for (const float* input : inputs)
                if (outputs[out] == input) return false;
            for (int prior = 0; prior < out; ++prior)
                if (outputs[out] == outputs[prior]) return false;
        }

        const std::size_t profile_bytes = static_cast<std::size_t>(d.nz) * sizeof(float);
        id<MTLBuffer> in[6] = {};
        for (int i = 0; i < 6; ++i)
        {
            in[i] = [impl_->hardware.device newBufferWithBytes:inputs[i]
                                                        length:i == 5 ? profile_bytes : bytes
                                                       options:MTLResourceStorageModeShared];
            if (in[i] == nil) return false;
        }
        id<MTLBuffer> out[4] = {};
        out[0] = [impl_->hardware.device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
        for (int i = 1; i < 4; ++i)
        {
            out[i] = d.accumulate_mass_flux
                ? [impl_->hardware.device newBufferWithBytes:outputs[i]
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared]
                : [impl_->hardware.device newBufferWithLength:bytes
                                                       options:MTLResourceStorageModeShared];
        }
        for (auto buffer : out) if (buffer == nil) return false;

        AcousticAgridContinuityConstants c{};
        c.nx = d.nx; c.ny = d.ny; c.nz = d.nz;
        c.dx = d.dx; c.dy = d.dy; c.dz = d.dz;
        c.dt_small = d.dt_small;
        c.lateral_bc_west = d.lateral_bc_west;
        c.lateral_bc_east = d.lateral_bc_east;
        c.lateral_bc_south = d.lateral_bc_south;
        c.lateral_bc_north = d.lateral_bc_north;
        c.top_bc = d.top_bc;
        c.accumulate_mass_flux = d.accumulate_mass_flux ? 1 : 0;
        c.mass_flux_weight = d.mass_flux_weight;
        c.frame_velocity_u_ms = d.frame_velocity_u_ms;
        c.frame_velocity_v_ms = d.frame_velocity_v_ms;

        id<MTLCommandBuffer> cb = impl_->command_buffer();
        id<MTLComputeCommandEncoder> encoder = cb ? [cb computeCommandEncoder] : nil;
        const NSUInteger maximum = impl_->pipeline(
            KernelId::AcousticContinuityMassFluxCartesianAgrid)
            .maxTotalThreadsPerThreadgroup;
        if (encoder == nil || maximum == 0) return false;
        [encoder setComputePipelineState:impl_->pipeline(
            KernelId::AcousticContinuityMassFluxCartesianAgrid)];
        for (NSUInteger i = 0; i < 6; ++i)
            [encoder setBuffer:in[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 4; ++i)
            [encoder setBuffer:out[i] offset:0 atIndex:i + 6];
        [encoder setBytes:&c length:sizeof(c) atIndex:10];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(cb)) return false;
        for (int i = 0; i < 4; ++i)
            std::memcpy(outputs[i], out[i].contents, bytes);
        ++impl_->telemetry.completed_acoustic_agrid_continuity_dispatches;
        return true;
    }
}
