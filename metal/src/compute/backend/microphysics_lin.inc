// Lin pointwise/sedimentation/fused execution and state application.
bool MetalComputeBackend::supports_lin_pointwise() const noexcept
{
    return ready(nimbus::compute::KernelId::LinPointwise);
}

bool MetalComputeBackend::supports_lin_sedimentation() const noexcept
{
    return ready(nimbus::compute::KernelId::LinSedimentation);
}

bool MetalComputeBackend::dispatch_lin_pointwise(const LinPointwiseDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0, field_bytes = 0;
        if (!supports_lin_pointwise() || !valid_lin_pointwise(dispatch, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
            return false;
        const float*  host_inputs[9]  = {dispatch.theta_data,
                                         dispatch.p_data,
                                         dispatch.qv,
                                         dispatch.qc,
                                         dispatch.qr,
                                         dispatch.qi,
                                         dispatch.qs,
                                         dispatch.qg,
                                         dispatch.qh};
        float*        host_outputs[8] = {dispatch.dtheta_dt,
                                         dispatch.dqv_dt,
                                         dispatch.dqc_dt,
                                         dispatch.dqr_dt,
                                         dispatch.dqi_dt,
                                         dispatch.dqs_dt,
                                         dispatch.dqg_dt,
                                         dispatch.dqh_dt};
        id<MTLBuffer> inputs[9]       = {};
        id<MTLBuffer> outputs[8]      = {};
        for (int i = 0; i < 9; ++i)
        {
            inputs[i] = [impl_->hardware.device newBufferWithBytes:host_inputs[i]
                                                            length:field_bytes
                                                           options:MTLResourceStorageModeShared];
            if (inputs[i] == nil)
                return false;
        }
        for (int i = 0; i < 8; ++i)
        {
            outputs[i] = [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
            if (outputs[i] == nil)
                return false;
        }
        LinPointwiseConstants        constants = lin_pointwise_constants(dispatch);
        id<MTLCommandBuffer>         command   = impl_->command_buffer();
        id<MTLComputeCommandEncoder> encoder   = command == nil ? nil : [command computeCommandEncoder];
        if (encoder == nil)
            return false;
        NSUInteger maximum = impl_->pipeline(KernelId::LinPointwise).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::LinPointwise)];
        for (NSUInteger i = 0; i < 9; ++i)
            [encoder setBuffer:inputs[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 8; ++i)
            [encoder setBuffer:outputs[i] offset:0 atIndex:i + 9];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:17];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
            return false;
        std::array<const float*, 8> result{};
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<const float*>(outputs[i].contents);
        if (!nimbus::compute::lin::finite_result(dispatch, result))
            return false;
        for (int i = 0; i < 8; ++i)
            std::memcpy(host_outputs[i], outputs[i].contents, field_bytes);
        ++impl_->telemetry.completed_lin_pointwise_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_lin_sedimentation(const LinSedimentationDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0, field_bytes = 0, columns = 0, ignored = 0;
        if (!supports_lin_sedimentation() || !valid_lin_sedimentation(dispatch, cells, field_bytes) ||
            !checked_bytes(dispatch.nx, dispatch.ny, 1, columns, ignored) || field_bytes > impl_->max_buffer_bytes())
            return false;
        const float*  host_inputs[6]  = {dispatch.qr,
                                         dispatch.qs,
                                         dispatch.qg,
                                         dispatch.qh,
                                         dispatch.rho_d,
                                         dispatch.layer_thickness_m ? dispatch.layer_thickness_m : dispatch.rho_d};
        float*        host_outputs[4] = {dispatch.dqr_dt, dispatch.dqs_dt, dispatch.dqg_dt, dispatch.dqh_dt};
        id<MTLBuffer> inputs[6]       = {};
        id<MTLBuffer> outputs[4]      = {};
        for (int i = 0; i < 6; ++i)
        {
            if (i == 5 && !dispatch.layer_thickness_m)
            {
                inputs[i] = inputs[4];
                continue;
            }
            inputs[i] = [impl_->hardware.device newBufferWithBytes:host_inputs[i]
                                                            length:field_bytes
                                                           options:MTLResourceStorageModeShared];
            if (inputs[i] == nil)
                return false;
        }
        for (int i = 0; i < 4; ++i)
        {
            outputs[i] = [impl_->hardware.device newBufferWithBytes:host_outputs[i]
                                                             length:field_bytes
                                                            options:MTLResourceStorageModeShared];
            if (outputs[i] == nil)
                return false;
        }
        LinSedimentationConstants    constants = lin_sedimentation_constants(dispatch);
        id<MTLCommandBuffer>         command   = impl_->command_buffer();
        id<MTLComputeCommandEncoder> encoder   = command == nil ? nil : [command computeCommandEncoder];
        if (encoder == nil)
            return false;
        NSUInteger maximum = impl_->pipeline(KernelId::LinSedimentation).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::LinSedimentation)];
        for (NSUInteger i = 0; i < 6; ++i)
            [encoder setBuffer:inputs[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 4; ++i)
            [encoder setBuffer:outputs[i] offset:0 atIndex:i + 6];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:10];
        [encoder dispatchThreads:MTLSizeMake(columns, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command))
            return false;
        const std::array<const float*, 4> result{static_cast<const float*>(outputs[0].contents),
                                                 static_cast<const float*>(outputs[1].contents),
                                                 static_cast<const float*>(outputs[2].contents),
                                                 static_cast<const float*>(outputs[3].contents)};
        if (!nimbus::compute::lin::finite_result(dispatch, result))
            return false;
        for (int i = 0; i < 4; ++i)
            std::memcpy(host_outputs[i], outputs[i].contents, field_bytes);
        ++impl_->telemetry.completed_lin_sedimentation_dispatches;
        return true;
    }
}

bool MetalComputeBackend::supports_atomic_lin_fused() const
{
    return supports(MicrophysicsKernelOp::LinFused);
}

bool MetalComputeBackend::dispatch_lin_fused(const LinFusedDispatch& dispatch)
{
    @autoreleasepool
    {
        const LinPointwiseDispatch&     point    = dispatch.pointwise;
        const LinSedimentationDispatch& sediment = dispatch.sedimentation;
        std::size_t                     cells = 0, field_bytes = 0, columns = 0, ignored = 0;
        if (!supports(MicrophysicsKernelOp::LinFused) ||
            !nimbus::compute::lin::valid_lin_fused(dispatch, cells, field_bytes) ||
            !checked_bytes(point.nx, point.ny, 1, columns, ignored) || field_bytes > impl_->max_buffer_bytes())
            return false;

        nimbus::compute::DispatchGraph schedule;
        const auto pointwise_node     = schedule.add(nimbus::compute::KernelId::LinPointwise,
                                                     {{"theta", nimbus::compute::ResourceAccess::Read},
                                                      {"pressure", nimbus::compute::ResourceAccess::Read},
                                                      {"qv", nimbus::compute::ResourceAccess::Read},
                                                      {"qc", nimbus::compute::ResourceAccess::Read},
                                                      {"qr", nimbus::compute::ResourceAccess::Read},
                                                      {"qi", nimbus::compute::ResourceAccess::Read},
                                                      {"qs", nimbus::compute::ResourceAccess::Read},
                                                      {"qg", nimbus::compute::ResourceAccess::Read},
                                                      {"qh", nimbus::compute::ResourceAccess::Read},
                                                      {"dtheta_dt", nimbus::compute::ResourceAccess::Write},
                                                      {"dqv_dt", nimbus::compute::ResourceAccess::Write},
                                                      {"dqc_dt", nimbus::compute::ResourceAccess::Write},
                                                      {"dqr_dt", nimbus::compute::ResourceAccess::Write},
                                                      {"dqi_dt", nimbus::compute::ResourceAccess::Write},
                                                      {"dqs_dt", nimbus::compute::ResourceAccess::Write},
                                                      {"dqg_dt", nimbus::compute::ResourceAccess::Write},
                                                      {"dqh_dt", nimbus::compute::ResourceAccess::Write}});
        const auto sedimentation_node = schedule.add(nimbus::compute::KernelId::LinSedimentation,
                                                     {{"qr", nimbus::compute::ResourceAccess::Read},
                                                      {"qs", nimbus::compute::ResourceAccess::Read},
                                                      {"qg", nimbus::compute::ResourceAccess::Read},
                                                      {"qh", nimbus::compute::ResourceAccess::Read},
                                                      {"rho_d", nimbus::compute::ResourceAccess::Read},
                                                      {"dqr_dt", nimbus::compute::ResourceAccess::ReadWrite},
                                                      {"dqs_dt", nimbus::compute::ResourceAccess::ReadWrite},
                                                      {"dqg_dt", nimbus::compute::ResourceAccess::ReadWrite},
                                                      {"dqh_dt", nimbus::compute::ResourceAccess::ReadWrite}});
        if (!schedule.valid() || !schedule.depends_on(sedimentation_node, pointwise_node) ||
            !schedule.requires_barrier(pointwise_node, sedimentation_node))
            return false;

        const float*  host_inputs[11] = {point.theta_data,
                                         point.p_data,
                                         point.qv,
                                         point.qc,
                                         point.qr,
                                         point.qi,
                                         point.qs,
                                         point.qg,
                                         point.qh,
                                         sediment.rho_d,
                                         sediment.layer_thickness_m ? sediment.layer_thickness_m : sediment.rho_d};
        float*        host_outputs[8] = {point.dtheta_dt,
                                         point.dqv_dt,
                                         point.dqc_dt,
                                         point.dqr_dt,
                                         point.dqi_dt,
                                         point.dqs_dt,
                                         point.dqg_dt,
                                         point.dqh_dt};
        id<MTLBuffer> inputs[11]      = {};
        for (int i = 0; i < 11; ++i)
        {
            if (i == 10 && !sediment.layer_thickness_m)
            {
                inputs[i] = inputs[9];
                continue;
            }
            inputs[i] = [impl_->hardware.device newBufferWithBytes:host_inputs[i]
                                                            length:field_bytes
                                                           options:MTLResourceStorageModeShared];
            if (inputs[i] == nil)
                return false;
        }
        static const char* const names[8] = {"lin_dtheta_dt",
                                             "lin_dqv_dt",
                                             "lin_dqc_dt",
                                             "lin_dqr_dt",
                                             "lin_dqi_dt",
                                             "lin_dqs_dt",
                                             "lin_dqg_dt",
                                             "lin_dqh_dt"};
        const std::size_t        sizes[8] = {
            field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, field_bytes};
        Impl::ResidentAllocation allocations[8];
        if (!impl_->resident.fields.acquire_many(names, sizes, 8, allocations))
            return false;
        Impl::ResidentSlot* slots[8]      = {};
        id<MTLBuffer>       tendencies[8] = {};
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        for (int i = 0; i < 8; ++i)
        {
            slots[i] = impl_->resident_slot(allocations[i]);
            if (slots[i] == nullptr || slots[i]->candidate == nil || slots[i]->current == nil ||
                slots[i]->bytes < field_bytes)
                return false;
            tendencies[i] = slots[i]->candidate;
            // Reserve version metadata before any host output can be published.
            impl_->resident.fields.commit_host_mirror(names[i], host_outputs[i], extent);
            impl_->resident.fields.invalidate(names[i]);
        }

        LinPointwiseConstants     point_constants = lin_pointwise_constants(point);
        LinSedimentationConstants sed_constants   = lin_sedimentation_constants(sediment);
        id<MTLCommandBuffer>      command         = impl_->command_buffer();
        if (command == nil)
            return false;
        id<MTLComputeCommandEncoder> point_encoder = [command computeCommandEncoder];
        if (point_encoder == nil)
            return false;
        NSUInteger point_max = impl_->pipeline(KernelId::LinPointwise).maxTotalThreadsPerThreadgroup;
        if (point_max == 0)
        {
            [point_encoder endEncoding];
            return false;
        }
        [point_encoder setComputePipelineState:impl_->pipeline(KernelId::LinPointwise)];
        for (NSUInteger i = 0; i < 9; ++i)
            [point_encoder setBuffer:inputs[i] offset:0 atIndex:i];
        for (NSUInteger i = 0; i < 8; ++i)
            [point_encoder setBuffer:tendencies[i] offset:0 atIndex:i + 9];
        [point_encoder setBytes:&point_constants length:sizeof(point_constants) atIndex:17];
        [point_encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, point_max), 1, 1)];
        [point_encoder endEncoding];

        id<MTLComputeCommandEncoder> sed_encoder = [command computeCommandEncoder];
        if (sed_encoder == nil)
            return false;
        NSUInteger sed_max = impl_->pipeline(KernelId::LinSedimentation).maxTotalThreadsPerThreadgroup;
        if (sed_max == 0)
        {
            [sed_encoder endEncoding];
            return false;
        }
        [sed_encoder setComputePipelineState:impl_->pipeline(KernelId::LinSedimentation)];
        [sed_encoder setBuffer:inputs[4] offset:0 atIndex:0];
        [sed_encoder setBuffer:inputs[6] offset:0 atIndex:1];
        [sed_encoder setBuffer:inputs[7] offset:0 atIndex:2];
        [sed_encoder setBuffer:inputs[8] offset:0 atIndex:3];
        [sed_encoder setBuffer:inputs[9] offset:0 atIndex:4];
        [sed_encoder setBuffer:inputs[10] offset:0 atIndex:5];
        [sed_encoder setBuffer:tendencies[3] offset:0 atIndex:6];
        [sed_encoder setBuffer:tendencies[5] offset:0 atIndex:7];
        [sed_encoder setBuffer:tendencies[6] offset:0 atIndex:8];
        [sed_encoder setBuffer:tendencies[7] offset:0 atIndex:9];
        [sed_encoder setBytes:&sed_constants length:sizeof(sed_constants) atIndex:10];
        [sed_encoder dispatchThreads:MTLSizeMake(columns, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, sed_max), 1, 1)];
        [sed_encoder endEncoding];
        if (!impl_->finish(command))
            return false;

        std::array<const float*, 8> result{};
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<const float*>(tendencies[i].contents);
        if (!nimbus::compute::lin::finite_result(point, result))
            return false;
        for (int i = 0; i < 8; ++i)
        {
            std::swap(slots[i]->current, slots[i]->candidate);
            std::memcpy(host_outputs[i], slots[i]->current.contents, field_bytes);
            impl_->resident.fields.note_device_write(names[i], host_outputs[i], extent);
            impl_->resident.fields.note_host_synced(names[i]);
        }
        ++impl_->telemetry.completed_lin_pointwise_dispatches;
        ++impl_->telemetry.completed_lin_sedimentation_dispatches;
        ++impl_->telemetry.completed_lin_fused_dispatches;
        impl_->telemetry.lin_resident_tendency_writes += 8;
        return true;
    }
}

bool MetalComputeBackend::supports_microphysics_apply() const noexcept
{
    return ready(nimbus::compute::KernelId::MicrophysicsApply);
}

bool MetalComputeBackend::dispatch_microphysics_apply(const MicrophysicsApplyDispatch& dispatch)
{
    @autoreleasepool
    {
        float* state_data[8] = {
            dispatch.theta_data, dispatch.qv_data, dispatch.qc_data, dispatch.qr_data,
            dispatch.qi_data, dispatch.qs_data, dispatch.qg_data, dispatch.qh_data};
        const float* tendency_data[8] = {
            dispatch.dtheta, dispatch.dqv, dispatch.dqc, dispatch.dqr,
            dispatch.dqi, dispatch.dqs, dispatch.dqg, dispatch.dqh};
        const float* auxiliary_data[3] = {
            dispatch.dtheta_rad, dispatch.dtheta_pbl, dispatch.dqv_pbl};
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        const std::int64_t stats_needed_64 =
            INT64_C(10) + static_cast<std::int64_t>(dispatch.nz) + static_cast<std::int64_t>(dispatch.nx);
        if (!supports_microphysics_apply() ||
            dispatch.i_begin < 0 || dispatch.i_begin >= dispatch.i_end || dispatch.i_end > dispatch.nx ||
            dispatch.j_begin < 0 || dispatch.j_begin >= dispatch.j_end || dispatch.j_end > dispatch.ny ||
            !std::isfinite(dispatch.dt_step) || dispatch.dt_step <= 0.0f ||
            !std::isfinite(dispatch.theta_fallback) ||
            !std::isfinite(dispatch.theta_min) || !std::isfinite(dispatch.theta_max) ||
            dispatch.theta_min > dispatch.theta_max ||
            !std::isfinite(dispatch.max_theta_step) || dispatch.max_theta_step < 0.0f ||
            !std::isfinite(dispatch.qv_max_val) || dispatch.qv_max_val < 0.0f ||
            !std::isfinite(dispatch.hydro_max_val) || dispatch.hydro_max_val < 0.0f ||
            stats_needed_64 <= 0 || stats_needed_64 > std::numeric_limits<int>::max() ||
            dispatch.stats_out == nullptr || dispatch.stats_len < static_cast<int>(stats_needed_64) ||
            !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        const std::size_t stats_needed = static_cast<std::size_t>(stats_needed_64);
        if (stats_needed > std::numeric_limits<std::size_t>::max() / sizeof(unsigned int))
            return false;
        const std::size_t stats_bytes = stats_needed * sizeof(unsigned int);
        if (stats_bytes > impl_->max_buffer_bytes())
            return false;

        for (float* state : state_data)
            if (state == nullptr)
                return false;
        for (const float* tendency : tendency_data)
            if (tendency == nullptr)
                return false;
        for (const float* auxiliary : auxiliary_data)
            if (auxiliary == nullptr)
                return false;
        const void* stats_pointer = static_cast<const void*>(dispatch.stats_out);
        for (int state = 0; state < 8; ++state)
        {
            for (int prior = 0; prior < state; ++prior)
                if (state_data[state] == state_data[prior])
                    return false;
            for (const float* tendency : tendency_data)
                if (state_data[state] == tendency)
                    return false;
            for (const float* auxiliary : auxiliary_data)
                if (state_data[state] == auxiliary)
                    return false;
            if (static_cast<const void*>(state_data[state]) == stats_pointer)
                return false;
        }
        for (const float* tendency : tendency_data)
            if (static_cast<const void*>(tendency) == stats_pointer)
                return false;
        for (const float* auxiliary : auxiliary_data)
            if (static_cast<const void*>(auxiliary) == stats_pointer)
                return false;

        static const char* const state_names[8] = {
            "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"};
        const std::size_t state_sizes[8] = {
            field_bytes, field_bytes, field_bytes, field_bytes,
            field_bytes, field_bytes, field_bytes, field_bytes};
        Impl::ResidentAllocation state_allocations[8];
        if (!impl_->resident.fields.acquire_many(state_names, state_sizes, 8, state_allocations))
            return false;
        Impl::ResidentSlot* state_slots[8] = {};
        id<MTLBuffer> state_buffers[8] = {};
        for (int index = 0; index < 8; ++index)
        {
            state_slots[index] = impl_->resident_slot(state_allocations[index]);
            if (state_slots[index] == nullptr || state_slots[index]->candidate == nil ||
                state_slots[index]->current == nil || state_slots[index]->bytes < field_bytes)
                return false;
            state_buffers[index] = state_slots[index]->candidate;
            std::memcpy(state_buffers[index].contents, state_data[index], field_bytes);
        }

        static const char* const resident_tendency_names[2][8] = {
            {"lin_dtheta_dt", "lin_dqv_dt", "lin_dqc_dt", "lin_dqr_dt",
             "lin_dqi_dt", "lin_dqs_dt", "lin_dqg_dt", "lin_dqh_dt"},
            {"kessler_dtheta_dt", "kessler_dqv_dt", "kessler_dqc_dt", "kessler_dqr_dt",
             nullptr, nullptr, "kessler_dqg_dt", "kessler_dqh_dt"}};
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        id<MTLBuffer> tendency_buffers[8] = {};
        std::uint64_t resident_reuses = 0;
        for (int index = 0; index < 8; ++index)
        {
            for (int family = 0; family < 2 && tendency_buffers[index] == nil; ++family)
            {
                const char* name = resident_tendency_names[family][index];
                if (name != nullptr && impl_->resident.fields.reusable(name, tendency_data[index], extent, true))
                {
                    const auto* entry = impl_->resident.fields.find(name);
                    Impl::ResidentSlot* slot = entry == nullptr ? nullptr : impl_->resident_slot(entry->allocation);
                    if (slot != nullptr && slot->current != nil && slot->bytes >= field_bytes)
                    {
                        tendency_buffers[index] = slot->current;
                        ++resident_reuses;
                    }
                }
            }
            if (tendency_buffers[index] == nil)
            {
                tendency_buffers[index] = [impl_->hardware.device newBufferWithBytes:tendency_data[index]
                                                                  length:field_bytes
                                                                 options:MTLResourceStorageModeShared];
                if (tendency_buffers[index] == nil)
                    return false;
            }
        }
        id<MTLBuffer> auxiliary_buffers[3] = {};
        for (int index = 0; index < 3; ++index)
        {
            auxiliary_buffers[index] = [impl_->hardware.device newBufferWithBytes:auxiliary_data[index]
                                                               length:field_bytes
                                                              options:MTLResourceStorageModeShared];
            if (auxiliary_buffers[index] == nil)
                return false;
        }
        id<MTLBuffer> stats_buffer =
            [impl_->hardware.device newBufferWithLength:stats_bytes options:MTLResourceStorageModeShared];
        if (stats_buffer == nil)
            return false;
        std::memset(stats_buffer.contents, 0, stats_bytes);

        MicrophysicsApplyConstants constants{};
        constants.nx = dispatch.nx;
        constants.ny = dispatch.ny;
        constants.nz = dispatch.nz;
        constants.i_begin = dispatch.i_begin;
        constants.i_end = dispatch.i_end;
        constants.j_begin = dispatch.j_begin;
        constants.j_end = dispatch.j_end;
        constants.dt_step = dispatch.dt_step;
        constants.theta_fallback = dispatch.theta_fallback;
        constants.theta_min = dispatch.theta_min;
        constants.theta_max = dispatch.theta_max;
        constants.max_theta_step = dispatch.max_theta_step;
        constants.qv_max = dispatch.qv_max_val;
        constants.hydro_max = dispatch.hydro_max_val;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::MicrophysicsApply).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::MicrophysicsApply)];
        for (NSUInteger index = 0; index < 8; ++index)
            [encoder setBuffer:state_buffers[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 8; ++index)
            [encoder setBuffer:tendency_buffers[index] offset:0 atIndex:index + 8];
        for (NSUInteger index = 0; index < 3; ++index)
            [encoder setBuffer:auxiliary_buffers[index] offset:0 atIndex:index + 16];
        [encoder setBuffer:stats_buffer offset:0 atIndex:19];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:20];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        for (int index = 0; index < 8; ++index)
            std::swap(state_slots[index]->current, state_slots[index]->candidate);
        for (int index = 0; index < 8; ++index)
        {
            std::memcpy(state_data[index], state_slots[index]->current.contents, field_bytes);
            impl_->resident.fields.note_device_write(state_names[index], state_data[index], extent);
            impl_->resident.fields.note_host_synced(state_names[index]);
        }
        std::memcpy(dispatch.stats_out, stats_buffer.contents, stats_bytes);
        ++impl_->telemetry.completed_microphysics_apply_dispatches;
        impl_->telemetry.microphysics_apply_resident_tendency_reuses += resident_reuses;
        return true;
    }
}
