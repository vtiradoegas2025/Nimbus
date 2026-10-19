// Full Cartesian tendency dispatch.
bool MetalComputeBackend::dispatch_cartesian_tendencies(const CartesianTendenciesDispatch& d)
{
    @autoreleasepool
    {
        std::size_t  cells = 0, bytes = 0;
        const float* inputs[13] = {d.u_x_data,
                                   d.u_y_data,
                                   d.w_data,
                                   d.rho_data,
                                   d.p_data,
                                   d.theta_data,
                                   d.p0_base_data,
                                   d.rho0_base_data,
                                   d.loading_data,
                                   d.u0_base_data,
                                   d.v0_base_data,
                                   d.qv_data,
                                   d.qv0_base_data};
        float*       outputs[5] = {d.du_x_dt_data, d.du_y_dt_data, d.dw_dt_data, d.drho_dt_data, d.dp_dt_data};
        if (!supports(DynamicsKernelOp::CartesianTendencies) || d.nr < 3 || d.nth < 3 || d.nz < 3 ||
            !checked_bytes(d.nr, d.nth, d.nz, cells, bytes) || !std::isfinite(d.dx) || d.dx <= 0 ||
            !std::isfinite(d.dy) || d.dy <= 0 || !std::isfinite(d.dz) || d.dz <= 0 || !std::isfinite(d.g) ||
            !std::isfinite(d.gamma_val) || d.gamma_val <= 0 || !std::isfinite(d.coriolis_f_val) ||
            !std::isfinite(d.coriolis_beta) ||
            bytes > impl_->max_buffer_bytes())
            return false;
        for (const float* input : inputs)
            if (input == nullptr)
                return false;
        for (int o = 0; o < 5; ++o)
        {
            if (outputs[o] == nullptr)
                return false;
            for (int p = 0; p < o; ++p)
                if (outputs[o] == outputs[p])
                    return false;
            for (const float* input : inputs)
                if (outputs[o] == input)
                    return false;
        }
        for (std::size_t n = 0; n < cells; ++n)
        {
            if (!std::isfinite(d.u_x_data[n]) || !std::isfinite(d.u_y_data[n]) ||
                !std::isfinite(d.w_data[n]) || !std::isfinite(d.rho_data[n]) ||
                d.rho_data[n] <= 0.0f || !std::isfinite(d.p_data[n]) ||
                !std::isfinite(d.theta_data[n]) || !std::isfinite(d.loading_data[n]) ||
                d.loading_data[n] < 0.0f ||
                (d.has_moisture && (!std::isfinite(d.qv_data[n]) || d.qv_data[n] < 0.0f)))
                return false;
        }
        for (int k = 0; k < d.nz; ++k)
        {
            if (!std::isfinite(d.p0_base_data[k]) || !std::isfinite(d.rho0_base_data[k]) ||
                d.rho0_base_data[k] <= 0.0f || !std::isfinite(d.u0_base_data[k]) ||
                !std::isfinite(d.v0_base_data[k]) ||
                (d.has_moisture && (!std::isfinite(d.qv0_base_data[k]) || d.qv0_base_data[k] < 0.0f)))
                return false;
        }
        const std::size_t profile_bytes = static_cast<std::size_t>(d.nz) * sizeof(float);
        const float*      ordered[13]   = {d.u_x_data,
                                           d.u_y_data,
                                           d.w_data,
                                           d.rho_data,
                                           d.p_data,
                                           d.theta_data,
                                           d.p0_base_data,
                                           d.rho0_base_data,
                                           d.loading_data,
                                           d.u0_base_data,
                                           d.v0_base_data,
                                           d.qv_data,
                                           d.qv0_base_data};
        const bool profile[13] = {false, false, false, false, false, false, true, true, false, true, true, false, true};
        id<MTLBuffer> in[13]   = {};
        for (int i = 0; i < 13; ++i)
        {
            in[i] = [impl_->hardware.device newBufferWithBytes:ordered[i]
                                                        length:profile[i] ? profile_bytes : bytes
                                                       options:MTLResourceStorageModeShared];
            if (in[i] == nil)
                return false;
        }
        static const char* const names[5] = {"slow_du_dt", "slow_dv_dt", "slow_dw_dt", "slow_drho_dt", "slow_dp_dt"};
        const std::size_t        sizes[5] = {bytes, bytes, bytes, bytes, bytes};
        Impl::ResidentAllocation allocations[5];
        if (!impl_->resident.fields.acquire_many(names, sizes, 5, allocations))
            return false;
        Impl::ResidentSlot* slots[5] = {};
        id<MTLBuffer>       out[5]   = {};
        for (int i = 0; i < 5; ++i)
        {
            slots[i] = impl_->resident_slot(allocations[i]);
            if (!slots[i] || slots[i]->candidate == nil)
                return false;
            out[i] = slots[i]->candidate;
        }
        CartesianAgridConstants c{};
        c.nr                                 = d.nr;
        c.nth                                = d.nth;
        c.nz                                 = d.nz;
        c.dx                                 = d.dx;
        c.dy                                 = d.dy;
        c.dz                                 = d.dz;
        c.g                                  = d.g;
        c.gamma_val                          = d.gamma_val;
        c.coriolis_f                         = d.coriolis_f_val;
        c.coriolis_beta                      = d.coriolis_beta;
        c.lateral_bc_west                    = d.lateral_bc_west;
        c.lateral_bc_east                    = d.lateral_bc_east;
        c.lateral_bc_south                   = d.lateral_bc_south;
        c.lateral_bc_north                   = d.lateral_bc_north;
        c.top_bc                             = d.top_bc;
        c.has_moisture                       = (d.has_moisture ? 1 : 0) |
                                               (d.split_slow_only ? 2 : 0);
        id<MTLCommandBuffer>         command = impl_->command_buffer();
        id<MTLComputeCommandEncoder> e       = command ? [command computeCommandEncoder] : nil;
        if (!e)
            return false;
        NSUInteger maximum = impl_->pipeline(KernelId::CartesianTendencies).maxTotalThreadsPerThreadgroup;
        if (!maximum)
        {
            [e endEncoding];
            return false;
        }
        [e setComputePipelineState:impl_->pipeline(KernelId::CartesianTendencies)];
        for (NSUInteger i = 0; i < 8; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 5; ++i)
            [e setBuffer:out[i] offset:0 atIndex:i + 8];
        for (NSUInteger i = 8; i < 13; ++i)
            [e setBuffer:in[i] offset:0 atIndex:i + 5];
        [e setBytes:&c length:sizeof(c) atIndex:18];
        [e dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [e endEncoding];
        if (!impl_->finish(command))
            return false;
        for (int i = 0; i < 5; ++i)
        {
            const float* candidate = static_cast<const float*>(out[i].contents);
            for (std::size_t n = 0; n < cells; ++n)
                if (!std::isfinite(candidate[n])) return false;
        }
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        for (int i = 0; i < 5; ++i)
        {
            std::swap(slots[i]->current, slots[i]->candidate);
            std::memcpy(outputs[i], slots[i]->current.contents, bytes);
            impl_->resident.fields.note_device_write(names[i], outputs[i], extent);
            impl_->resident.fields.note_host_synced(names[i]);
        }
        ++impl_->telemetry.completed_cartesian_agrid_tendency_dispatches;
        return true;
    }
}
