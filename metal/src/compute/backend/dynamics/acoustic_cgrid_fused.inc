// One fused Cartesian C-grid acoustic substep.
bool MetalComputeBackend::supports_unrepaired_acoustic_fused() const
{
    return supports(DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid);
}

bool MetalComputeBackend::dispatch_acoustic_substep_fused_cartesian_cgrid(
    const AcousticSubstepFusedCartesianCgridDispatch& d)
{
    @autoreleasepool
    {
        if (!d.repair_primary_state && !valid_acoustic_cgrid_dispatch(d))
            return false;
        std::size_t cells = 0, bytes = 0;
        if (!supports(DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid) || d.energy_report_host != nullptr ||
            !d.u || !d.v || !d.w || !d.rho || !d.p || !d.theta_data || !d.rho0_base_data || !d.p0_base_data ||
            !d.dp0_dz_data || d.base_len < d.nz || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || !std::isfinite(d.dx) || d.dx <= 0 ||
            !std::isfinite(d.dy) || d.dy <= 0 || !std::isfinite(d.dz) || d.dz <= 0 || !std::isfinite(d.gamma_val) ||
            d.gamma_val <= 0 || !std::isfinite(d.dt_small) || d.dt_small < 0 || !std::isfinite(d.rho_min) ||
            !std::isfinite(d.rho_max) || d.rho_min > d.rho_max || !std::isfinite(d.p_min) || !std::isfinite(d.p_max) ||
            d.p_min > d.p_max || !std::isfinite(d.p_ref_fallback) || !std::isfinite(d.wind_clamp_h) ||
            d.wind_clamp_h < 0 || !std::isfinite(d.wind_clamp_v) || d.wind_clamp_v < 0 ||
            !std::isfinite(d.frame_velocity_u_ms) || !std::isfinite(d.frame_velocity_v_ms))
            return false;
        void* state_data[] = {d.u, d.v, d.w, d.rho, d.p};
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < i; ++j)
                if (state_data[i] == state_data[j])
                    return false;
        static const char* const state_names[]  = {"u", "v", "w", "rho", "p"};
        std::size_t              state_sizes[]  = {bytes, bytes, bytes, bytes, bytes};
        Impl::ResidentAllocation state_alloc[5] = {};
        if (!impl_->resident.fields.acquire_many(state_names, state_sizes, 5, state_alloc))
            return false;
        Impl::ResidentSlot*                state[5] = {};
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        std::uint64_t                      uploads = 0, reuses = 0;
        for (int i = 0; i < 5; ++i)
        {
            state[i] = impl_->resident_slot(state_alloc[i]);
            if (!state[i] || state[i]->current == nil || state[i]->candidate == nil || state[i]->bytes < bytes)
                return false;
            if (impl_->resident.fields.reusable(state_names[i], state_data[i], extent, true))
                ++reuses;
            else
            {
                impl_->resident.fields.invalidate(state_names[i]);
                std::memcpy(state[i]->current.contents, state_data[i], bytes);
                ++uploads;
            }
            impl_->resident.fields.commit_host_mirror(state_names[i], state_data[i], extent);
            impl_->resident.fields.invalidate(state_names[i]);
        }
        AcousticMomentumCartesianCgridDispatch gradient_dispatch;
        gradient_dispatch.dp0_dz_data = d.dp0_dz_data;
        gradient_dispatch.dp0_dz_low_data = d.dp0_dz_low_data;
        gradient_dispatch.nz = d.nz;
        std::vector<float> paired_gradient;
        const auto* gradient = d.dp0_dz_low_data
            ? pack_acoustic_base_gradient(gradient_dispatch, paired_gradient) : d.dp0_dz_data;
        static const char* const inv_names[]  = {"acoustic_rho0", "acoustic_p0", "acoustic_dp0_dz", "acoustic_theta"};
        const void*              inv_data[]   = {d.rho0_base_data, d.p0_base_data, gradient, d.theta_data};
        const std::size_t        prof         = static_cast<std::size_t>(d.nz) * sizeof(float);
        std::size_t              inv_sizes[]  = {prof, prof, prof * (d.dp0_dz_low_data ? 2 : 1), bytes};
        Impl::ResidentAllocation inv_alloc[4] = {};
        if (!impl_->resident.fields.acquire_many(inv_names, inv_sizes, 4, inv_alloc))
            return false;
        Impl::ResidentSlot* inv[4]            = {};
        bool                inv_reused[4]     = {};
        std::uint64_t       invariant_uploads = 0, invariant_reuses = 0;
        for (int i = 0; i < 4; ++i)
        {
            inv[i] = impl_->resident_slot(inv_alloc[i]);
            nimbus::compute::FieldExtent ex{inv_sizes[i] / sizeof(float), sizeof(float)};
            if (!inv[i] || inv[i]->current == nil || inv[i]->bytes < inv_sizes[i])
                return false;
            inv_reused[i] = impl_->resident.fields.reusable(inv_names[i], inv_data[i], ex, true);
            if (inv_reused[i])
            {
                ++reuses;
                ++invariant_reuses;
            }
            else
            {
                impl_->resident.fields.invalidate(inv_names[i]);
                std::memcpy(inv[i]->current.contents, inv_data[i], inv_sizes[i]);
                ++uploads;
                ++invariant_uploads;
            }
            if (!inv_reused[i])
            {
                impl_->resident.fields.commit_host_mirror(inv_names[i], inv_data[i], ex);
                impl_->resident.fields.invalidate(inv_names[i]);
            }
        }
        id<MTLCommandBuffer>         cb = impl_->command_buffer();
        id<MTLComputeCommandEncoder> pe = [cb computeCommandEncoder];
        if (cb == nil || pe == nil)
            return false;
        AcousticPressureCgridConstants pc{};
        pc.nx                  = d.nx;
        pc.ny                  = d.ny;
        pc.nz                  = d.nz;
        pc.dx                  = d.dx;
        pc.dy                  = d.dy;
        pc.dz_val              = d.dz;
        pc.gamma_val           = d.gamma_val;
        pc.gamma_low           = d.repair_primary_state ? 0.0f : d.gamma_low;
        pc.preserve_unrepaired = d.repair_primary_state ? 0 : 1;
        pc.dt_small            = d.dt_small;
        pc.rho_min             = d.rho_min;
        pc.rho_max             = d.rho_max;
        pc.p_min               = d.p_min;
        pc.p_max               = d.p_max;
        pc.p_ref_fallback      = d.p_ref_fallback;
        pc.frame_velocity_u_ms = d.frame_velocity_u_ms;
        pc.frame_velocity_v_ms = d.frame_velocity_v_ms;
        [pe setComputePipelineState:impl_->pipeline(KernelId::AcousticPressureCartesianCgrid)];
        for (NSUInteger i = 0; i < 5; ++i)
            [pe setBuffer:state[i]->current offset:0 atIndex:i];
        [pe setBuffer:inv[0]->current offset:0 atIndex:5];
        [pe setBuffer:inv[1]->current offset:0 atIndex:6];
        [pe setBuffer:inv[3]->current offset:0 atIndex:7];
        [pe setBuffer:state[3]->candidate offset:0 atIndex:8];
        [pe setBuffer:state[4]->candidate offset:0 atIndex:9];
        [pe setBytes:&pc length:sizeof(pc) atIndex:10];
        [pe dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      std::min<NSUInteger>(64,
                                                           impl_->pipeline(KernelId::AcousticPressureCartesianCgrid)
                                                               .maxTotalThreadsPerThreadgroup),
                                      1,
                                      1)];
        [pe endEncoding];
        id<MTLComputeCommandEncoder> me = [cb computeCommandEncoder];
        if (me == nil)
            return false;
        AcousticMomentumCgridConstants mc{};
        mc.nx           = d.nx;
        mc.ny           = d.ny;
        mc.nz           = d.nz;
        mc.dx           = d.dx;
        mc.dy           = d.dy;
        mc.dz_val       = d.dz;
        mc.dt_small     = d.dt_small;
        mc.wind_clamp_h = d.wind_clamp_h;
        mc.wind_clamp_v = d.wind_clamp_v;
        mc.preserve_unrepaired = d.repair_primary_state ? 0 : 1;
        mc.paired_base_gradient = d.dp0_dz_low_data ? 1 : 0;
        [me setComputePipelineState:impl_->pipeline(KernelId::AcousticMomentumCartesianCgrid)];
        [me setBuffer:state[3]->candidate offset:0 atIndex:0];
        [me setBuffer:state[4]->candidate offset:0 atIndex:1];
        [me setBuffer:inv[2]->current offset:0 atIndex:2];
        for (NSUInteger i = 0; i < 3; ++i)
        {
            [me setBuffer:state[i]->current offset:0 atIndex:i + 3];
            [me setBuffer:state[i]->candidate offset:0 atIndex:i + 6];
        }
        [me setBytes:&mc length:sizeof(mc) atIndex:9];
        [me dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      std::min<NSUInteger>(64,
                                                           impl_->pipeline(KernelId::AcousticMomentumCartesianCgrid)
                                                               .maxTotalThreadsPerThreadgroup),
                                      1,
                                      1)];
        [me endEncoding];
        if (!impl_->finish(cb))
            return false;
        if (!d.repair_primary_state && !valid_unrepaired_acoustic_state({
                static_cast<const float*>(state[0]->candidate.contents), static_cast<const float*>(state[1]->candidate.contents),
                static_cast<const float*>(state[2]->candidate.contents), static_cast<const float*>(state[3]->candidate.contents),
                static_cast<const float*>(state[4]->candidate.contents)}, cells))
            return false;
        for (int i = 0; i < 5; ++i)
            std::swap(state[i]->current, state[i]->candidate);
        for (int i = 0; i < 5; ++i)
        {
            std::memcpy(state_data[i], state[i]->current.contents, bytes);
            impl_->resident.fields.note_device_write(state_names[i], state_data[i], extent);
            impl_->resident.fields.note_host_synced(state_names[i]);
        }
        for (int i = 0; i < 4; ++i)
        {
            nimbus::compute::FieldExtent ex{inv_sizes[i] / sizeof(float), sizeof(float)};
            impl_->resident.fields.commit_host_mirror(inv_names[i], inv_data[i], ex);
        }
        impl_->telemetry.resident_uploads += uploads;
        impl_->telemetry.resident_reuses += reuses;
        impl_->telemetry.acoustic_invariant_uploads += invariant_uploads;
        impl_->telemetry.acoustic_invariant_reuses += invariant_reuses;
        ++impl_->telemetry.completed_acoustic_pressure_dispatches;
        ++impl_->telemetry.completed_acoustic_momentum_dispatches;
        ++impl_->telemetry.completed_acoustic_fused_dispatches;
        return true;
    }
}

bool MetalComputeBackend::supports_batched_cartesian_cgrid_mass_flux_export() const
{
    return supports(DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid) &&
           ready(nimbus::compute::KernelId::AcousticPressureMassFluxCartesianCgrid);
}
