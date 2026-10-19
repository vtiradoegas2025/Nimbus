// State relaxation, slow tendencies, and atomic slow-state application.
bool MetalComputeBackend::dispatch_state_relaxation_cartesian(
    const StateRelaxationDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0, field_bytes = 0;
        const std::int64_t required_profiles =
            static_cast<std::int64_t>(dispatch.nz) * 7;
        if (!supports(DynamicsKernelOp::StateRelaxationCartesian) ||
            dispatch.u == nullptr || dispatch.v == nullptr || dispatch.w == nullptr ||
            dispatch.theta_data == nullptr || dispatch.profiles == nullptr ||
            !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            required_profiles <= 0 || dispatch.profiles_len < required_profiles ||
            dispatch.lateral_width < 0 || dispatch.lateral_width > dispatch.nz ||
            (!dispatch.do_upper && !dispatch.do_w_only && !dispatch.do_lateral) ||
            !std::isfinite(dispatch.dt_step) || dispatch.dt_step <= 0.0f ||
            field_bytes > impl_->max_buffer_bytes())
            return false;

        void* state[4] = {dispatch.u, dispatch.v, dispatch.w, dispatch.theta_data};
        for (int field = 0; field < 4; ++field)
        {
            if (state[field] == dispatch.profiles)
                return false;
            for (int prior = 0; prior < field; ++prior)
                if (state[field] == state[prior])
                    return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
            for (int field = 0; field < 4; ++field)
                if (!std::isfinite(static_cast<float*>(state[field])[index]))
                    return false;
        for (std::int64_t index = 0; index < required_profiles; ++index)
            if (!std::isfinite(dispatch.profiles[index]))
                return false;

        static const char* const names[4] = {"u", "v", "w", "theta"};
        const std::size_t sizes[4] = {field_bytes, field_bytes, field_bytes, field_bytes};
        Impl::ResidentAllocation allocations[4] = {};
        if (!impl_->resident.fields.acquire_many(names, sizes, 4, allocations))
            return false;
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        Impl::ResidentSlot* slots[4] = {};
        std::uint64_t uploads = 0, reuses = 0;
        for (int field = 0; field < 4; ++field)
        {
            slots[field] = impl_->resident_slot(allocations[field]);
            if (slots[field] == nullptr || slots[field]->current == nil ||
                slots[field]->candidate == nil || slots[field]->bytes < field_bytes)
                return false;
            if (impl_->resident.fields.reusable(names[field], state[field], extent, true))
                ++reuses;
            else
            {
                impl_->resident.fields.invalidate(names[field]);
                std::memcpy(slots[field]->current.contents, state[field], field_bytes);
                ++uploads;
            }
        }
        const std::size_t profile_bytes =
            static_cast<std::size_t>(required_profiles) * sizeof(float);
        id<MTLBuffer> profiles = [impl_->hardware.device newBufferWithBytes:dispatch.profiles
                                                           length:profile_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (profiles == nil || command_buffer == nil || encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::StateRelaxationCartesian).maxTotalThreadsPerThreadgroup;
        if (maximum == 0) { [encoder endEncoding]; return false; }
        StateRelaxationConstants constants{};
        constants.nx = dispatch.nx; constants.ny = dispatch.ny; constants.nz = dispatch.nz;
        constants.lateral_width = dispatch.lateral_width;
        constants.do_upper = dispatch.do_upper ? 1 : 0;
        constants.do_w_only = dispatch.do_w_only ? 1 : 0;
        constants.do_lateral = dispatch.do_lateral ? 1 : 0;
        constants.dt_step = dispatch.dt_step;
        [encoder setComputePipelineState:impl_->pipeline(KernelId::StateRelaxationCartesian)];
        for (NSUInteger field = 0; field < 4; ++field)
            [encoder setBuffer:slots[field]->current offset:0 atIndex:field];
        [encoder setBuffer:profiles offset:0 atIndex:4];
        for (NSUInteger field = 0; field < 4; ++field)
            [encoder setBuffer:slots[field]->candidate offset:0 atIndex:field + 5];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:9];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        if (dispatch.completed_observer != nullptr)
            dispatch.completed_observer(
                dispatch.completed_observer_context,
                static_cast<const float*>(slots[0]->candidate.contents),
                static_cast<const float*>(slots[1]->candidate.contents),
                static_cast<const float*>(slots[2]->candidate.contents),
                static_cast<const float*>(slots[3]->candidate.contents));
        for (int field = 0; field < 4; ++field)
            std::swap(slots[field]->current, slots[field]->candidate);
        for (int field = 0; field < 4; ++field)
        {
            std::memcpy(state[field], slots[field]->current.contents, field_bytes);
            impl_->resident.fields.note_device_write(names[field], state[field], extent);
            impl_->resident.fields.note_host_synced(names[field]);
        }
        impl_->telemetry.resident_uploads += uploads;
        impl_->telemetry.resident_reuses += reuses;
        impl_->telemetry.dynamics_state_resident_reuses += reuses;
        ++impl_->telemetry.completed_state_relaxation_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_cartesian_cgrid_slow_tendencies(
    const CartesianCgridSlowTendenciesDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0, field_bytes = 0;
        const bool export_advection = dispatch.du_advection_dt_data != nullptr ||
            dispatch.dv_advection_dt_data != nullptr || dispatch.dw_advection_dt_data != nullptr;
        if (!supports(DynamicsKernelOp::CartesianCgridSlowTendencies) ||
            dispatch.u_data == nullptr || dispatch.v_data == nullptr || dispatch.w_data == nullptr ||
            dispatch.rho_data == nullptr || dispatch.p_data == nullptr ||
            dispatch.rho0_base_data == nullptr || dispatch.rho0_base_len < dispatch.nz ||
            dispatch.du_dt_data == nullptr || dispatch.dv_dt_data == nullptr ||
            dispatch.dw_dt_data == nullptr || dispatch.drho_dt_data == nullptr ||
            dispatch.dp_dt_data == nullptr || dispatch.nr < 3 || dispatch.nth < 3 || dispatch.nz < 2 ||
            !std::isfinite(dispatch.dx) || dispatch.dx <= 0.0f ||
            !std::isfinite(dispatch.dy) || dispatch.dy <= 0.0f ||
            !std::isfinite(dispatch.dz) || dispatch.dz <= 0.0f ||
            !std::isfinite(dispatch.g) || !std::isfinite(dispatch.coriolis_f_val) ||
            !std::isfinite(dispatch.coriolis_beta) ||
            !std::isfinite(dispatch.frame_velocity_u_ms) ||
            !std::isfinite(dispatch.frame_velocity_v_ms) ||
            (dispatch.has_moisture && dispatch.qv_data == nullptr) ||
            (dispatch.has_loading && dispatch.loading_data == nullptr) ||
            (export_advection && (dispatch.du_advection_dt_data == nullptr ||
                dispatch.dv_advection_dt_data == nullptr || dispatch.dw_advection_dt_data == nullptr)) ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
            return false;

        const void* inputs[11] = {dispatch.u_data,dispatch.v_data,dispatch.w_data,
            dispatch.rho_data,dispatch.p_data,dispatch.rho0_base_data,dispatch.qv0_base_data,
            dispatch.u0_base_data,dispatch.v0_base_data,dispatch.qv_data,dispatch.loading_data};
        void* outputs[8] = {dispatch.du_dt_data,dispatch.dv_dt_data,dispatch.dw_dt_data,
            dispatch.drho_dt_data,dispatch.dp_dt_data,dispatch.du_advection_dt_data,
            dispatch.dv_advection_dt_data,dispatch.dw_advection_dt_data};
        const int output_count = export_advection ? 8 : 5;
        for (int output = 0; output < output_count; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs[output] == outputs[prior]) return false;
            for (const void* input : inputs)
                if (input != nullptr && outputs[output] == input) return false;
        }
        for (int k = 0; k < dispatch.nz; ++k)
            if (!std::isfinite(dispatch.rho0_base_data[k]) || dispatch.rho0_base_data[k] <= 0.0f)
                return false;

        std::vector<float> qv0_scratch(static_cast<std::size_t>(dispatch.nz),0.0f);
        std::vector<float> u0_scratch(static_cast<std::size_t>(dispatch.nz),0.0f);
        std::vector<float> v0_scratch(static_cast<std::size_t>(dispatch.nz),0.0f);
        const float* qv0_upload = qv0_scratch.data();
        if (dispatch.has_moisture && dispatch.qv0_base_data != nullptr)
        {
            const int count = std::max(0,std::min(dispatch.qv0_base_len,dispatch.nz));
            std::copy_n(dispatch.qv0_base_data,static_cast<std::size_t>(count),qv0_scratch.data());
        }
        const float* u0_upload = u0_scratch.data();
        const float* v0_upload = v0_scratch.data();
        if (dispatch.u0_base_data != nullptr && dispatch.v0_base_data != nullptr)
        {
            const int count = std::max(0,std::min(dispatch.wind0_base_len,dispatch.nz));
            std::copy_n(dispatch.u0_base_data,static_cast<std::size_t>(count),u0_scratch.data());
            std::copy_n(dispatch.v0_base_data,static_cast<std::size_t>(count),v0_scratch.data());
        }
        for (int k = 0; k < dispatch.nz; ++k)
            if (!std::isfinite(qv0_upload[k]) || !std::isfinite(u0_upload[k]) || !std::isfinite(v0_upload[k]))
                return false;

        id<MTLBuffer> input_buffers[11] = {};
        const float zero = 0.0f;
        const void* upload_data[11] = {dispatch.u_data,dispatch.v_data,dispatch.w_data,
            dispatch.rho_data,dispatch.p_data,dispatch.rho0_base_data,qv0_upload,u0_upload,v0_upload,
            dispatch.has_moisture?dispatch.qv_data:&zero,
            dispatch.has_loading?dispatch.loading_data:&zero};
        const std::size_t profile_bytes=static_cast<std::size_t>(dispatch.nz)*sizeof(float);
        const std::size_t upload_bytes[11] = {field_bytes,field_bytes,field_bytes,field_bytes,field_bytes,
            profile_bytes,profile_bytes,profile_bytes,profile_bytes,
            dispatch.has_moisture?field_bytes:sizeof(float),dispatch.has_loading?field_bytes:sizeof(float)};

        // The orchestration-owned Heun pair changes u/v/w/p between stages,
        // but rho and qv are invariant. Keep exactly those full fields
        // resident when explicitly requested; profiles and loading remain
        // transient to preserve the same memory tradeoff as Vulkan.
        const nimbus::compute::FieldExtent extent{cells,sizeof(float)};
        static const char* const invariant_names[2]={"rho","qv"};
        const void* invariant_sources[2]={dispatch.rho_data,dispatch.qv_data};
        const int invariant_input_indices[2]={3,9};
        const std::size_t invariant_count=dispatch.slow_rk2_invariant_input_residency
            ? (dispatch.has_moisture?2u:1u):0u;
        std::size_t invariant_sizes[2]={field_bytes,field_bytes};
        Impl::ResidentAllocation invariant_allocations[2]={};
        Impl::ResidentSlot* invariant_slots[2]={};
        std::uint64_t invariant_uploads=0,invariant_reuses=0;
        if(invariant_count!=0)
        {
            if(!impl_->resident.fields.acquire_many(invariant_names,invariant_sizes,
                                                    invariant_count,invariant_allocations))return false;
            for(std::size_t field=0;field<invariant_count;++field)
            {
                invariant_slots[field]=impl_->resident_slot(invariant_allocations[field]);
                if(invariant_slots[field]==nullptr||invariant_slots[field]->current==nil||
                   invariant_slots[field]->current.contents==nullptr||
                   invariant_slots[field]->bytes<field_bytes)return false;
                if(impl_->resident.fields.reusable(
                       invariant_names[field],invariant_sources[field],extent,true))
                    ++invariant_reuses;
                else
                {
                    impl_->resident.fields.invalidate(invariant_names[field]);
                    std::memcpy(invariant_slots[field]->current.contents,
                                invariant_sources[field],field_bytes);
                    ++invariant_uploads;
                }
                input_buffers[invariant_input_indices[field]]=invariant_slots[field]->current;
            }
        }
        for (int index=0;index<11;++index)
        {
            if(input_buffers[index]!=nil)continue;
            input_buffers[index]=[impl_->hardware.device newBufferWithBytes:upload_data[index]
                                                          length:upload_bytes[index]
                                                         options:MTLResourceStorageModeShared];
            if(input_buffers[index]==nil)return false;
        }

        static const char* const tendency_names[5]={"slow_du_dt","slow_dv_dt","slow_dw_dt","slow_drho_dt","slow_dp_dt"};
        const std::size_t sizes[5]={field_bytes,field_bytes,field_bytes,field_bytes,field_bytes};
        Impl::ResidentAllocation allocations[5]={};
        if(!impl_->resident.fields.acquire_many(tendency_names,sizes,5,allocations))return false;
        Impl::ResidentSlot* slots[5]={};
        for(int field=0;field<5;++field)
        {
            slots[field]=impl_->resident_slot(allocations[field]);
            if(slots[field]==nullptr||slots[field]->current==nil||slots[field]->bytes<field_bytes)return false;
            impl_->resident.fields.invalidate(tendency_names[field]);
        }
        id<MTLBuffer> advection_outputs[3]={};
        if(export_advection)
            for(int field=0;field<3;++field)
            {
                advection_outputs[field]=[impl_->hardware.device newBufferWithLength:field_bytes
                                                                    options:MTLResourceStorageModeShared];
                if(advection_outputs[field]==nil)return false;
            }

        id<MTLCommandBuffer> command_buffer=impl_->command_buffer();
        id<MTLComputeCommandEncoder> encoder=[command_buffer computeCommandEncoder];
        if(command_buffer==nil||encoder==nil)return false;
        const NSUInteger maximum=impl_->pipeline(KernelId::CartesianCgridSlowTendencies).maxTotalThreadsPerThreadgroup;
        if(maximum==0){[encoder endEncoding];return false;}
        CartesianCgridSlowConstants constants{};
        constants.nr=dispatch.nr;constants.nth=dispatch.nth;constants.nz=dispatch.nz;
        constants.dr=dispatch.dx;constants.dtheta=dispatch.dy;constants.dz_val=dispatch.dz;
        constants.g=dispatch.g;constants.has_moisture=dispatch.has_moisture?1:0;
        constants.has_loading=dispatch.has_loading?1:0;constants.padding0=dispatch.coriolis_f_val;
        constants.padding1=export_advection?1.0f:0.0f;
        constants.coriolis_beta=dispatch.coriolis_beta;
        constants.frame_velocity_u_ms=dispatch.frame_velocity_u_ms;
        constants.frame_velocity_v_ms=dispatch.frame_velocity_v_ms;
        [encoder setComputePipelineState:impl_->pipeline(KernelId::CartesianCgridSlowTendencies)];
        for(NSUInteger index=0;index<11;++index)[encoder setBuffer:input_buffers[index] offset:0 atIndex:index];
        for(NSUInteger index=0;index<5;++index)[encoder setBuffer:slots[index]->current offset:0 atIndex:index+11];
        for(NSUInteger index=0;index<3;++index)[encoder setBuffer:export_advection?advection_outputs[index]:slots[index]->current offset:0 atIndex:index+16];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:19];
        [encoder dispatchThreads:MTLSizeMake(cells,1,1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,maximum),1,1)];
        [encoder endEncoding];
        if(!impl_->finish(command_buffer))return false;

        for(std::size_t field=0;field<invariant_count;++field)
            impl_->resident.fields.commit_host_mirror(
                invariant_names[field],invariant_sources[field],extent);
        for(int field=0;field<5;++field)
        {
            std::memcpy(outputs[field],slots[field]->current.contents,field_bytes);
            impl_->resident.fields.note_device_write(tendency_names[field],outputs[field],extent);
            impl_->resident.fields.note_host_synced(tendency_names[field]);
        }
        if(export_advection)
            for(int field=0;field<3;++field)
                std::memcpy(outputs[field+5],advection_outputs[field].contents,field_bytes);
        impl_->telemetry.resident_uploads+=invariant_uploads;
        impl_->telemetry.resident_reuses+=invariant_reuses;
        impl_->telemetry.slow_tendency_invariant_input_uploads+=invariant_uploads;
        impl_->telemetry.slow_tendency_invariant_input_reuses+=invariant_reuses;
        ++impl_->telemetry.completed_cartesian_cgrid_slow_tendency_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_slow_apply(const SlowApplyDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0, field_bytes = 0;
        if (!supports(DynamicsKernelOp::SlowApply) || dispatch.u == nullptr ||
            dispatch.v == nullptr || dispatch.w == nullptr || dispatch.p_data == nullptr ||
            dispatch.du_dt == nullptr || dispatch.dv_dt == nullptr || dispatch.dw_dt == nullptr ||
            dispatch.dp_dt == nullptr || dispatch.du_dt_pbl == nullptr || dispatch.dv_dt_pbl == nullptr ||
            !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            !std::isfinite(dispatch.dt_large) || dispatch.dt_large <= 0.0f ||
            field_bytes > impl_->max_buffer_bytes())
            return false;

        void* state[4] = {dispatch.u, dispatch.v, dispatch.w, dispatch.p_data};
        const void* tendencies[6] = {dispatch.du_dt, dispatch.dv_dt, dispatch.dw_dt,
                                     dispatch.dp_dt, dispatch.du_dt_pbl, dispatch.dv_dt_pbl};
        for (int field = 0; field < 4; ++field)
        {
            for (int prior = 0; prior < field; ++prior)
                if (state[field] == state[prior])
                    return false;
            for (const void* tendency : tendencies)
                if (state[field] == tendency)
                    return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
            for (int field = 0; field < 4; ++field)
                if (!std::isfinite(static_cast<float*>(state[field])[index]))
                    return false;

        static const char* const names[4] = {"u", "v", "w", "p"};
        const std::size_t sizes[4] = {field_bytes, field_bytes, field_bytes, field_bytes};
        Impl::ResidentAllocation allocations[4] = {};
        if (!impl_->resident.fields.acquire_many(names, sizes, 4, allocations))
            return false;
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        Impl::ResidentSlot* slots[4] = {};
        std::uint64_t uploads = 0, reuses = 0;
        for (int field = 0; field < 4; ++field)
        {
            slots[field] = impl_->resident_slot(allocations[field]);
            if (slots[field] == nullptr || slots[field]->current == nil ||
                slots[field]->candidate == nil || slots[field]->bytes < field_bytes)
                return false;
            if (impl_->resident.fields.reusable(names[field], state[field], extent, true))
                ++reuses;
            else
            {
                impl_->resident.fields.invalidate(names[field]);
                std::memcpy(slots[field]->current.contents, state[field], field_bytes);
                ++uploads;
            }
        }
        id<MTLBuffer> tendency_buffers[6] = {};
        static const char* const resident_tendency_names[4] = {
            "slow_du_dt", "slow_dv_dt", "slow_dw_dt", "slow_dp_dt"};
        const int resident_source_indices[4] = {0, 1, 2, 3};
        std::uint64_t resident_tendency_reuses = 0;
        for (int field = 0; field < 4; ++field)
        {
            const int source = resident_source_indices[field];
            const auto* entry = impl_->resident.fields.find(resident_tendency_names[field]);
            auto* slot = entry == nullptr ? nullptr : impl_->resident_slot(entry->allocation);
            if (impl_->resident.fields.reusable(
                    resident_tendency_names[field], tendencies[source], extent, true) &&
                slot != nullptr && slot->current != nil && slot->bytes >= field_bytes)
            {
                tendency_buffers[source] = slot->current;
                ++resident_tendency_reuses;
            }
            else
            {
                tendency_buffers[source] = [impl_->hardware.device newBufferWithBytes:tendencies[source]
                                                                   length:field_bytes
                                                                  options:MTLResourceStorageModeShared];
            }
            if (tendency_buffers[source] == nil)
                return false;
        }
        for (int field = 4; field < 6; ++field)
        {
            tendency_buffers[field] = [impl_->hardware.device newBufferWithBytes:tendencies[field]
                                                               length:field_bytes
                                                              options:MTLResourceStorageModeShared];
            if (tendency_buffers[field] == nil)
                return false;
        }
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::SlowApply).maxTotalThreadsPerThreadgroup;
        if (maximum == 0) { [encoder endEncoding]; return false; }
        SlowApplyConstants constants{};
        constants.nx = dispatch.nx; constants.ny = dispatch.ny; constants.nz = dispatch.nz;
        constants.dt_large = dispatch.dt_large;
        constants.sanitize = dispatch.sanitize ? 1 : 0;
        [encoder setComputePipelineState:impl_->pipeline(KernelId::SlowApply)];
        for (NSUInteger field = 0; field < 4; ++field)
            [encoder setBuffer:slots[field]->current offset:0 atIndex:field];
        for (NSUInteger field = 0; field < 6; ++field)
            [encoder setBuffer:tendency_buffers[field] offset:0 atIndex:field + 4];
        for (NSUInteger field = 0; field < 4; ++field)
            [encoder setBuffer:slots[field]->candidate offset:0 atIndex:field + 10];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:14];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        if (dispatch.completed_observer != nullptr)
            dispatch.completed_observer(
                dispatch.completed_observer_context,
                static_cast<const float*>(slots[0]->candidate.contents),
                static_cast<const float*>(slots[1]->candidate.contents),
                static_cast<const float*>(slots[2]->candidate.contents),
                static_cast<const float*>(slots[3]->candidate.contents));
        for (int field = 0; field < 4; ++field)
            std::swap(slots[field]->current, slots[field]->candidate);
        for (int field = 0; field < 4; ++field)
        {
            std::memcpy(state[field], slots[field]->current.contents, field_bytes);
            impl_->resident.fields.note_device_write(names[field], state[field], extent);
            impl_->resident.fields.note_host_synced(names[field]);
        }
        impl_->telemetry.resident_uploads += uploads;
        impl_->telemetry.resident_reuses += reuses;
        impl_->telemetry.dynamics_state_resident_reuses += reuses;
        impl_->telemetry.slow_apply_resident_tendency_reuses += resident_tendency_reuses;
        ++impl_->telemetry.completed_slow_apply_dispatches;
        return true;
    }
}
