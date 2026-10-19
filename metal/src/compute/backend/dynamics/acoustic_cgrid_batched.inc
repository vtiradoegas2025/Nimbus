// Batched Cartesian C-grid acoustic loop with optional flux export.
bool MetalComputeBackend::dispatch_acoustic_substeps_batched_cartesian_cgrid(
    const AcousticSubstepsBatchedCartesianCgridDispatch& d)
{
    @autoreleasepool
    {
        if (!d.repair_primary_state || d.dp0_dz_low_data || d.wave_history || d.wave_history_output ||
            d.mass_flux_weight != 0.0 || d.gamma_low != 0.0f)
            return false;
        std::size_t cells = 0, bytes = 0;
        const bool  valid_lateral = d.bc_west >= 0 && d.bc_west <= 4 && d.bc_east >= 0 && d.bc_east <= 4 &&
                                    d.bc_south >= 0 && d.bc_south <= 4 && d.bc_north >= 0 && d.bc_north <= 4;
        if (!supports(DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid) ||
            (d.accumulate_mass_flux && !supports_batched_cartesian_cgrid_mass_flux_export()) || !d.u || !d.v || !d.w ||
            !d.rho || !d.p || !d.theta_data || !d.rho0_base_data || !d.p0_base_data || !d.dp0_dz_data ||
            d.base_len < d.nz || d.nx < 4 || d.ny < 4 || d.nz < 2 || d.n_substeps < 1 || d.n_substeps > 4096 ||
            !valid_lateral || d.bc_bottom < 0 || d.bc_bottom > 1 || d.bc_top < 0 || d.bc_top > 2 ||
            (d.accumulate_mass_flux && (!d.mass_flux_x || !d.mass_flux_y || !d.mass_flux_z)) ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || !std::isfinite(d.dx) || d.dx <= 0 ||
            !std::isfinite(d.dy) || d.dy <= 0 || !std::isfinite(d.dz) || d.dz <= 0 || !std::isfinite(d.gamma_val) ||
            d.gamma_val <= 0 || !std::isfinite(d.dt_small) || d.dt_small < 0 || !std::isfinite(d.rho_min) ||
            !std::isfinite(d.rho_max) || d.rho_min > d.rho_max || !std::isfinite(d.p_min) || !std::isfinite(d.p_max) ||
            d.p_min > d.p_max || !std::isfinite(d.p_ref_fallback) || !std::isfinite(d.wind_clamp_h) ||
            d.wind_clamp_h < 0 || !std::isfinite(d.wind_clamp_v) || d.wind_clamp_v < 0 ||
            !std::isfinite(d.frame_velocity_u_ms) || !std::isfinite(d.frame_velocity_v_ms))
            return false;
        void* host_state[] = {d.u, d.v, d.w, d.rho, d.p};
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < i; ++j)
                if (host_state[i] == host_state[j])
                    return false;
        if (d.accumulate_mass_flux)
        {
            void* flux[] = {d.mass_flux_x, d.mass_flux_y, d.mass_flux_z};
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < i; ++j)
                    if (flux[i] == flux[j])
                        return false;
                for (void* s : host_state)
                    if (flux[i] == s)
                        return false;
            }
        }

        static const char* const state_names[]  = {"u", "v", "w", "rho", "p"};
        std::size_t              state_sizes[]  = {bytes, bytes, bytes, bytes, bytes};
        Impl::ResidentAllocation state_alloc[5] = {};
        if (!impl_->resident.fields.acquire_many(state_names, state_sizes, 5, state_alloc))
            return false;
        Impl::ResidentSlot*                state[5] = {};
        std::array<id<MTLBuffer>, 5>       cur{}, next{};
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        std::uint64_t                      uploads = 0, reuses = 0;
        for (int i = 0; i < 5; ++i)
        {
            state[i] = impl_->resident_slot(state_alloc[i]);
            if (!state[i] || state[i]->current == nil || state[i]->candidate == nil || state[i]->bytes < bytes)
                return false;
            bool reuse = impl_->resident.fields.reusable(state_names[i], host_state[i], extent, true);
            if (reuse)
                ++reuses;
            else
            {
                std::memcpy(state[i]->current.contents, host_state[i], bytes);
                ++uploads;
            }
            cur[i]  = state[i]->current;
            next[i] = state[i]->candidate;
            impl_->resident.fields.invalidate(state_names[i]);
        }

        const std::size_t        prof         = static_cast<std::size_t>(d.nz) * sizeof(float);
        static const char* const inv_names[]  = {"acoustic_rho0", "acoustic_p0", "acoustic_dp0_dz", "acoustic_theta"};
        const void*              inv_data[]   = {d.rho0_base_data, d.p0_base_data, d.dp0_dz_data, d.theta_data};
        std::size_t              inv_sizes[]  = {prof, prof, prof, bytes};
        Impl::ResidentAllocation inv_alloc[4] = {};
        if (!impl_->resident.fields.acquire_many(inv_names, inv_sizes, 4, inv_alloc))
            return false;
        Impl::ResidentSlot* inv[4]      = {};
        std::uint64_t       inv_uploads = 0, inv_reuses = 0;
        for (int i = 0; i < 4; ++i)
        {
            inv[i] = impl_->resident_slot(inv_alloc[i]);
            nimbus::compute::FieldExtent ex{i == 3 ? cells : static_cast<std::size_t>(d.nz), sizeof(float)};
            if (!inv[i] || inv[i]->current == nil || inv[i]->bytes < inv_sizes[i])
                return false;
            if (impl_->resident.fields.reusable(inv_names[i], inv_data[i], ex, true))
            {
                ++reuses;
                ++inv_reuses;
            }
            else
            {
                impl_->resident.fields.invalidate(inv_names[i]);
                std::memcpy(inv[i]->current.contents, inv_data[i], inv_sizes[i]);
                ++uploads;
                ++inv_uploads;
            }
        }

        const std::size_t rad_elements =
            2u * (static_cast<std::size_t>(d.nx) + static_cast<std::size_t>(d.ny)) * static_cast<std::size_t>(d.nz);
        id<MTLBuffer> rad = [impl_->hardware.device newBufferWithLength:rad_elements * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
        if (rad == nil)
            return false;
        static const char* const flux_names[]  = {"cgrid_mass_flux_x", "cgrid_mass_flux_y", "cgrid_mass_flux_z"};
        Impl::ResidentSlot*      flux_slots[3] = {};
        Impl::ResidentAllocation flux_alloc[3] = {};
        if (d.accumulate_mass_flux)
        {
            std::size_t fs[] = {bytes, bytes, bytes};
            if (!impl_->resident.fields.acquire_many(flux_names, fs, 3, flux_alloc))
                return false;
            for (int i = 0; i < 3; ++i)
            {
                flux_slots[i] = impl_->resident_slot(flux_alloc[i]);
                if (!flux_slots[i] || flux_slots[i]->current == nil || flux_slots[i]->bytes < bytes)
                    return false;
                std::memset(flux_slots[i]->current.contents, 0, bytes);
                impl_->resident.fields.invalidate(flux_names[i]);
            }
        }

        id<MTLCommandBuffer> cb = impl_->command_buffer();
        if (cb == nil)
            return false;
        const NSUInteger               width = 64;
        AcousticPressureCgridConstants pc{};
        pc.nx                  = d.nx;
        pc.ny                  = d.ny;
        pc.nz                  = d.nz;
        pc.dx                  = d.dx;
        pc.dy                  = d.dy;
        pc.dz_val              = d.dz;
        pc.gamma_val           = d.gamma_val;
        pc.dt_small            = d.dt_small;
        pc.rho_min             = d.rho_min;
        pc.rho_max             = d.rho_max;
        pc.p_min               = d.p_min;
        pc.p_max               = d.p_max;
        pc.p_ref_fallback      = d.p_ref_fallback;
        pc.pad0                = d.accumulate_mass_flux ? 1.0f / static_cast<float>(d.n_substeps) : 0.0f;
        pc.frame_velocity_u_ms = d.frame_velocity_u_ms;
        pc.frame_velocity_v_ms = d.frame_velocity_v_ms;
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
        AcousticBcFaceConstants bc{};
        bc.nx                  = d.nx;
        bc.ny                  = d.ny;
        bc.nz                  = d.nz;
        bc.bc_west             = d.bc_west;
        bc.bc_east             = d.bc_east;
        bc.bc_south            = d.bc_south;
        bc.bc_north            = d.bc_north;
        bc.bc_bottom           = d.bc_bottom;
        bc.bc_top              = d.bc_top;
        bc.frame_velocity_u_ms = d.frame_velocity_u_ms;
        bc.frame_velocity_v_ms = d.frame_velocity_v_ms;
        bc.dt_small            = d.dt_small;
        bc.inv_dx              = 1.0f / d.dx;
        bc.inv_dy              = 1.0f / d.dy;
        auto encode_bc         = [&](const std::array<id<MTLBuffer>, 5>& fields, int face) -> bool
        {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            if (e == nil)
                return false;
            bc.face = face;
            [e setComputePipelineState:impl_->pipeline(KernelId::AcousticBcFaceCartesianCgrid)];
            for (NSUInteger i = 0; i < 5; ++i)
                [e setBuffer:fields[i] offset:0 atIndex:i];
            [e setBuffer:rad offset:0 atIndex:5];
            [e setBytes:&bc length:sizeof(bc) atIndex:6];
            [e dispatchThreads:MTLSizeMake(cells, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                                          std::min<NSUInteger>(width,
                                                               impl_->pipeline(KernelId::AcousticBcFaceCartesianCgrid)
                                                                   .maxTotalThreadsPerThreadgroup),
                                          1,
                                          1)];
            [e endEncoding];
            return true;
        };
        if (!encode_bc(cur, 5))
            return false;
        for (int step = 0; step < d.n_substeps; ++step)
        {
            id<MTLComputeCommandEncoder> pe = [cb computeCommandEncoder];
            if (pe == nil)
                return false;
            id<MTLComputePipelineState> pp = d.accumulate_mass_flux
                                                 ? impl_->pipeline(KernelId::AcousticPressureMassFluxCartesianCgrid)
                                                 : impl_->pipeline(KernelId::AcousticPressureCartesianCgrid);
            [pe setComputePipelineState:pp];
            for (NSUInteger i = 0; i < 5; ++i)
                [pe setBuffer:cur[i] offset:0 atIndex:i];
            [pe setBuffer:inv[0]->current offset:0 atIndex:5];
            [pe setBuffer:inv[1]->current offset:0 atIndex:6];
            [pe setBuffer:inv[3]->current offset:0 atIndex:7];
            [pe setBuffer:next[3] offset:0 atIndex:8];
            [pe setBuffer:next[4] offset:0 atIndex:9];
            if (d.accumulate_mass_flux)
                for (NSUInteger i = 0; i < 3; ++i)
                    [pe setBuffer:flux_slots[i]->current offset:0 atIndex:i + 10];
            [pe setBytes:&pc length:sizeof(pc) atIndex:d.accumulate_mass_flux ? 13 : 10];
            [pe dispatchThreads:MTLSizeMake(cells, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(width, pp.maxTotalThreadsPerThreadgroup), 1, 1)];
            [pe endEncoding];
            id<MTLComputeCommandEncoder> me = [cb computeCommandEncoder];
            if (me == nil)
                return false;
            [me setComputePipelineState:impl_->pipeline(KernelId::AcousticMomentumCartesianCgrid)];
            [me setBuffer:next[3] offset:0 atIndex:0];
            [me setBuffer:next[4] offset:0 atIndex:1];
            [me setBuffer:inv[2]->current offset:0 atIndex:2];
            for (NSUInteger i = 0; i < 3; ++i)
            {
                [me setBuffer:cur[i] offset:0 atIndex:i + 3];
                [me setBuffer:next[i] offset:0 atIndex:i + 6];
            }
            [me setBytes:&mc length:sizeof(mc) atIndex:9];
            [me dispatchThreads:MTLSizeMake(cells, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                                          std::min<NSUInteger>(width,
                                                               impl_->pipeline(KernelId::AcousticMomentumCartesianCgrid)
                                                                   .maxTotalThreadsPerThreadgroup),
                                          1,
                                          1)];
            [me endEncoding];
            for (int face = 0; face < 5; ++face)
                if (!encode_bc(next, face))
                    return false;
            for (int i = 0; i < 5; ++i)
                std::swap(cur[i], next[i]);
        }
        if (!impl_->finish(cb))
            return false;
        for (int i = 0; i < 5; ++i)
        {
            if (state[i]->current != cur[i])
                std::swap(state[i]->current, state[i]->candidate);
            std::memcpy(host_state[i], state[i]->current.contents, bytes);
            impl_->resident.fields.note_device_write(state_names[i], host_state[i], extent);
            impl_->resident.fields.note_host_synced(state_names[i]);
        }
        for (int i = 0; i < 4; ++i)
        {
            nimbus::compute::FieldExtent ex{i == 3 ? cells : static_cast<std::size_t>(d.nz), sizeof(float)};
            impl_->resident.fields.commit_host_mirror(inv_names[i], inv_data[i], ex);
        }
        if (d.accumulate_mass_flux)
        {
            void* out[] = {d.mass_flux_x, d.mass_flux_y, d.mass_flux_z};
            for (int i = 0; i < 3; ++i)
            {
                std::memcpy(out[i], flux_slots[i]->current.contents, bytes);
                impl_->resident.fields.note_device_write(flux_names[i], out[i], extent);
                impl_->resident.fields.note_host_synced(flux_names[i]);
            }
            ++impl_->telemetry.completed_acoustic_mass_flux_exports;
            impl_->telemetry.completed_acoustic_mass_flux_pressure_dispatches +=
                static_cast<std::uint64_t>(d.n_substeps);
        }
        impl_->telemetry.resident_uploads += uploads;
        impl_->telemetry.resident_reuses += reuses;
        impl_->telemetry.acoustic_invariant_uploads += inv_uploads;
        impl_->telemetry.acoustic_invariant_reuses += inv_reuses;
        impl_->telemetry.completed_acoustic_pressure_dispatches += static_cast<std::uint64_t>(d.n_substeps);
        impl_->telemetry.completed_acoustic_momentum_dispatches += static_cast<std::uint64_t>(d.n_substeps);
        impl_->telemetry.completed_acoustic_boundary_face_dispatches +=
            1u + 5u * static_cast<std::uint64_t>(d.n_substeps);
        ++impl_->telemetry.completed_acoustic_batched_dispatches;
        return true;
    }
}

void MetalComputeBackend::release_acoustic_loop_residency()
{
    for (const char* name : {"acoustic_rho0", "acoustic_p0", "acoustic_dp0_dz", "acoustic_theta",
                              "u", "v", "w", "rho", "p"})
        impl_->resident.fields.release(name);
}
