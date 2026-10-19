// Smagorinsky coefficient/tendency graphs and SGS state application.
TurbulenceComputeKernels* MetalComputeBackend::turbulence_kernels()
{
    return supports_smagorinsky_fused() ? this : nullptr;
}

const TurbulenceComputeKernels* MetalComputeBackend::turbulence_kernels() const
{
    return supports_smagorinsky_fused() ? this : nullptr;
}

bool MetalComputeBackend::supports(TurbulenceKernelOp op) const
{
    switch (op)
    {
    case TurbulenceKernelOp::Smagorinsky:
        return supports_smagorinsky_fused();
    case TurbulenceKernelOp::CollocatedSgs:
        return ready(KernelId::CollocatedSgs);
    case TurbulenceKernelOp::Tke:
        return ready(KernelId::TkeClosure) && ready(KernelId::TkeTendency) && supports_smagorinsky_tendencies() &&
               supports_sgs_apply();
    case TurbulenceKernelOp::SgsApply:
        return supports_sgs_apply();
    }
    return false;
}

void MetalComputeBackend::begin_resident_step()
{
    impl_->resident.fields.begin_step();
}

void MetalComputeBackend::invalidate_resident_field(const char* field_id) noexcept
{
    if (field_id != nullptr)
        impl_->resident.fields.note_host_write(field_id);
}

void MetalComputeBackend::invalidate_all_resident_fields() noexcept
{
    impl_->resident.fields.invalidate_all();
}

bool MetalComputeBackend::supports_smagorinsky_coefficients() const noexcept
{
    return ready(nimbus::compute::KernelId::SmagorinskyCoefficients);
}

bool MetalComputeBackend::dispatch_smagorinsky_coefficients(const SmagorinskyCoefficientsDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells       = 0;
        std::size_t field_bytes = 0;
        if (!supports_smagorinsky_coefficients() || dispatch.u == nullptr || dispatch.v == nullptr ||
            dispatch.w == nullptr || dispatch.dz_prof == nullptr || dispatch.dzc_prof == nullptr ||
            dispatch.nu_t == nullptr || dispatch.K_theta == nullptr || dispatch.K_q == nullptr ||
            dispatch.prof_len < dispatch.nz || dispatch.filter_mode < 0 || dispatch.filter_mode > 1 ||
            (!dispatch.use_horizontal && !dispatch.use_vertical) || !std::isfinite(dispatch.dx) ||
            !std::isfinite(dispatch.dy) || dispatch.dx <= 0.0f || dispatch.dy <= 0.0f || !std::isfinite(dispatch.Cs) ||
            dispatch.Cs < 0.0f || !std::isfinite(dispatch.Pr_t) || dispatch.Pr_t <= 0.0f ||
            !std::isfinite(dispatch.Sc_t) || dispatch.Sc_t <= 0.0f || !std::isfinite(dispatch.nu_t_max) ||
            dispatch.nu_t_max < 0.0f || !std::isfinite(dispatch.K_max) || dispatch.K_max < 0.0f ||
            !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            (dispatch.periodic_x && dispatch.nx < 3) || (dispatch.periodic_y && dispatch.ny < 3) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        const std::size_t profile_bytes = static_cast<std::size_t>(dispatch.nz) * sizeof(float);
        for (int k = 0; k < dispatch.nz; ++k)
        {
            if (!std::isfinite(dispatch.dz_prof[k]) || dispatch.dz_prof[k] <= 0.0f ||
                !std::isfinite(dispatch.dzc_prof[k]) || dispatch.dzc_prof[k] <= 0.0f)
            {
                return false;
            }
        }

        const void* inputs[5]  = {dispatch.u, dispatch.v, dispatch.w, dispatch.dz_prof, dispatch.dzc_prof};
        const void* outputs[3] = {dispatch.nu_t, dispatch.K_theta, dispatch.K_q};
        for (int output = 0; output < 3; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs[output] == outputs[prior])
                    return false;
            for (const void* input : inputs)
                if (outputs[output] == input)
                    return false;
        }

        id<MTLBuffer> u         = [impl_->hardware.device newBufferWithBytes:dispatch.u
                                                             length:field_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> v         = [impl_->hardware.device newBufferWithBytes:dispatch.v
                                                             length:field_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> w         = [impl_->hardware.device newBufferWithBytes:dispatch.w
                                                             length:field_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> dz        = [impl_->hardware.device newBufferWithBytes:dispatch.dz_prof
                                                             length:profile_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> dzc       = [impl_->hardware.device newBufferWithBytes:dispatch.dzc_prof
                                                             length:profile_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> viscosity = [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> theta_diffusivity    = [impl_->hardware.device newBufferWithLength:field_bytes
                                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> moisture_diffusivity = [impl_->hardware.device newBufferWithLength:field_bytes
                                                                        options:MTLResourceStorageModeShared];
        if (u == nil || v == nil || w == nil || dz == nil || dzc == nil || viscosity == nil ||
            theta_diffusivity == nil || moisture_diffusivity == nil)
        {
            return false;
        }

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;

        SmagorinskyConstants constants{};
        constants.nx             = dispatch.nx;
        constants.ny             = dispatch.ny;
        constants.nz             = dispatch.nz;
        constants.filter_mode    = dispatch.filter_mode;
        constants.use_horizontal = dispatch.use_horizontal ? 1 : 0;
        constants.use_vertical   = dispatch.use_vertical ? 1 : 0;
        constants.dx             = dispatch.dx;
        constants.dy             = dispatch.dy;
        constants.Cs             = dispatch.Cs;
        constants.Pr_t           = dispatch.Pr_t;
        constants.Sc_t           = dispatch.Sc_t;
        constants.nu_t_max       = dispatch.nu_t_max;
        constants.K_max          = dispatch.K_max;
        constants.periodic_y     = dispatch.periodic_y ? 1 : 0;
        constants.periodic_x     = dispatch.periodic_x ? 1 : 0;

        [encoder setComputePipelineState:impl_->pipeline(KernelId::SmagorinskyCoefficients)];
        [encoder setBuffer:u offset:0 atIndex:0];
        [encoder setBuffer:v offset:0 atIndex:1];
        [encoder setBuffer:w offset:0 atIndex:2];
        [encoder setBuffer:dz offset:0 atIndex:3];
        [encoder setBuffer:dzc offset:0 atIndex:4];
        [encoder setBuffer:viscosity offset:0 atIndex:5];
        [encoder setBuffer:theta_diffusivity offset:0 atIndex:6];
        [encoder setBuffer:moisture_diffusivity offset:0 atIndex:7];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:8];

        const NSUInteger maximum = impl_->pipeline(KernelId::SmagorinskyCoefficients).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        const NSUInteger width = std::min<NSUInteger>(64, maximum);
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];

        if (!impl_->finish(command_buffer))
            return false;

        std::memcpy(dispatch.nu_t, viscosity.contents, field_bytes);
        std::memcpy(dispatch.K_theta, theta_diffusivity.contents, field_bytes);
        std::memcpy(dispatch.K_q, moisture_diffusivity.contents, field_bytes);
        ++impl_->telemetry.completed_smagorinsky_coefficients_dispatches;
        return true;
    }
}

bool MetalComputeBackend::supports_smagorinsky_tendencies() const noexcept
{
    return ready(nimbus::compute::KernelId::SmagorinskyTendencies);
}

bool MetalComputeBackend::dispatch_smagorinsky_tendencies(const SmagorinskyTendenciesDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells       = 0;
        std::size_t field_bytes = 0;
        if (!supports_smagorinsky_tendencies() || dispatch.u == nullptr || dispatch.v == nullptr ||
            dispatch.w == nullptr || dispatch.theta_data == nullptr || dispatch.qv_data == nullptr ||
            dispatch.nu_t == nullptr || dispatch.K_theta == nullptr || dispatch.K_q == nullptr ||
            dispatch.dz_prof == nullptr || dispatch.rho == nullptr || dispatch.dudt_sgs == nullptr ||
            dispatch.dvdt_sgs == nullptr || dispatch.dwdt_sgs == nullptr || dispatch.dthetadt_sgs == nullptr ||
            dispatch.dqvdt_sgs == nullptr || dispatch.prof_len < dispatch.nz ||
            (!dispatch.use_horizontal && !dispatch.use_vertical) || !std::isfinite(dispatch.dx) ||
            !std::isfinite(dispatch.dy) || dispatch.dx <= 0.0f || dispatch.dy <= 0.0f ||
            !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            dispatch.nx < 3 || dispatch.ny < 3 || dispatch.nz < 2 ||
            (dispatch.periodic_x && dispatch.nx < 3) || (dispatch.periodic_y && dispatch.ny < 3) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        const std::size_t profile_bytes = static_cast<std::size_t>(dispatch.nz) * sizeof(float);
        for (int k = 0; k < dispatch.nz; ++k)
            if (!std::isfinite(dispatch.dz_prof[k]) || dispatch.dz_prof[k] <= 0.0f)
                return false;

        const float* field_inputs[9] = {dispatch.u,       dispatch.v,       dispatch.w,
                                        dispatch.theta_data, dispatch.qv_data, dispatch.nu_t,
                                        dispatch.K_theta, dispatch.K_q,     dispatch.rho};
        for (std::size_t index = 0; index < cells; ++index)
        {
            if (!std::isfinite(dispatch.rho[index]) || dispatch.rho[index] <= 0.0f ||
                !std::isfinite(dispatch.nu_t[index]) || dispatch.nu_t[index] < 0.0f ||
                !std::isfinite(dispatch.K_theta[index]) || dispatch.K_theta[index] < 0.0f ||
                !std::isfinite(dispatch.K_q[index]) || dispatch.K_q[index] < 0.0f)
            {
                return false;
            }
            for (int input = 0; input < 5; ++input)
                if (!std::isfinite(field_inputs[input][index]))
                    return false;
        }

        const void* inputs[10] = {dispatch.u,       dispatch.v,       dispatch.w,      dispatch.theta_data,
                                  dispatch.qv_data, dispatch.nu_t,    dispatch.K_theta, dispatch.K_q,
                                  dispatch.dz_prof, dispatch.rho};
        const void* outputs[5] = {dispatch.dudt_sgs, dispatch.dvdt_sgs, dispatch.dwdt_sgs,
                                  dispatch.dthetadt_sgs, dispatch.dqvdt_sgs};
        for (int output = 0; output < 5; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs[output] == outputs[prior])
                    return false;
            for (const void* input : inputs)
                if (outputs[output] == input)
                    return false;
        }

        id<MTLBuffer> input_buffers[10] = {
            [impl_->hardware.device newBufferWithBytes:dispatch.u length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.v length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.w length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.theta_data length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.qv_data length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.nu_t length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.K_theta length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.K_q length:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.dz_prof length:profile_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithBytes:dispatch.rho length:field_bytes options:MTLResourceStorageModeShared],
        };
        id<MTLBuffer> output_buffers[5] = {
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
        };
        for (id<MTLBuffer> buffer : input_buffers)
            if (buffer == nil)
                return false;
        for (id<MTLBuffer> buffer : output_buffers)
            if (buffer == nil)
                return false;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;

        SmagorinskyTendenciesConstants constants{};
        constants.nx             = dispatch.nx;
        constants.ny             = dispatch.ny;
        constants.nz             = dispatch.nz;
        constants.use_horizontal = dispatch.use_horizontal ? 1 : 0;
        constants.use_vertical   = dispatch.use_vertical ? 1 : 0;
        constants.has_qv         = dispatch.has_qv ? 1 : 0;
        constants.periodic_x     = dispatch.periodic_x ? 1 : 0;
        constants.periodic_y     = dispatch.periodic_y ? 1 : 0;
        constants.dx             = dispatch.dx;
        constants.dy             = dispatch.dy;
        constants.closed_lateral = turbulence_closed_lateral_mask(dispatch.open_lateral);

        [encoder setComputePipelineState:impl_->pipeline(KernelId::SmagorinskyTendencies)];
        for (NSUInteger index = 0; index < 10; ++index)
            [encoder setBuffer:input_buffers[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 5; ++index)
            [encoder setBuffer:output_buffers[index] offset:0 atIndex:index + 10];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:15];

        const NSUInteger maximum = impl_->pipeline(KernelId::SmagorinskyTendencies).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        const NSUInteger width = std::min<NSUInteger>(64, maximum);
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];

        if (!impl_->finish(command_buffer))
            return false;

        void* destinations[5] = {dispatch.dudt_sgs, dispatch.dvdt_sgs, dispatch.dwdt_sgs,
                                 dispatch.dthetadt_sgs, dispatch.dqvdt_sgs};
        for (int output = 0; output < 5; ++output)
            std::memcpy(destinations[output], output_buffers[output].contents, field_bytes);
        ++impl_->telemetry.completed_smagorinsky_tendencies_dispatches;
        return true;
    }
}

bool MetalComputeBackend::supports_smagorinsky_fused() const noexcept
{
    return supports_smagorinsky_coefficients() && supports_smagorinsky_tendencies();
}

bool MetalComputeBackend::dispatch_smagorinsky_fused(const SmagorinskyFusedDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells       = 0;
        std::size_t field_bytes = 0;
        if (!supports_smagorinsky_fused() || dispatch.u_data == nullptr || dispatch.v_data == nullptr ||
            dispatch.w_data == nullptr || dispatch.theta_data == nullptr || dispatch.qv_data == nullptr ||
            dispatch.dz_prof == nullptr || dispatch.dzc_prof == nullptr || dispatch.rho == nullptr ||
            dispatch.dudt_sgs == nullptr || dispatch.dvdt_sgs == nullptr || dispatch.dwdt_sgs == nullptr ||
            dispatch.dthetadt_sgs == nullptr || dispatch.dqvdt_sgs == nullptr || dispatch.prof_len < dispatch.nz ||
            (dispatch.download_coefficients &&
             (dispatch.nu_t == nullptr || dispatch.K_theta == nullptr || dispatch.K_q == nullptr)) ||
            dispatch.filter_mode < 0 || dispatch.filter_mode > 1 ||
            (!dispatch.use_horizontal && !dispatch.use_vertical) || !std::isfinite(dispatch.dx_val) ||
            !std::isfinite(dispatch.dy_val) || dispatch.dx_val <= 0.0f || dispatch.dy_val <= 0.0f ||
            !std::isfinite(dispatch.Cs) || dispatch.Cs < 0.0f || !std::isfinite(dispatch.Pr_t) ||
            dispatch.Pr_t <= 0.0f || !std::isfinite(dispatch.Sc_t) || dispatch.Sc_t <= 0.0f ||
            !std::isfinite(dispatch.nu_t_max) || dispatch.nu_t_max < 0.0f || !std::isfinite(dispatch.K_max) ||
            dispatch.K_max < 0.0f || !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            dispatch.nx < 3 || dispatch.ny < 3 || dispatch.nz < 2 ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        const std::size_t profile_bytes = static_cast<std::size_t>(dispatch.nz) * sizeof(float);
        for (int k = 0; k < dispatch.nz; ++k)
            if (!std::isfinite(dispatch.dz_prof[k]) || dispatch.dz_prof[k] <= 0.0f ||
                !std::isfinite(dispatch.dzc_prof[k]) || dispatch.dzc_prof[k] <= 0.0f)
                return false;
        const float* fields[6] = {dispatch.u_data, dispatch.v_data, dispatch.w_data,
                                  dispatch.theta_data, dispatch.qv_data, dispatch.rho};
        for (std::size_t index = 0; index < cells; ++index)
        {
            for (const float* field : fields)
                if (!std::isfinite(field[index]))
                    return false;
            if (dispatch.rho[index] <= 0.0f)
                return false;
        }

        const void* inputs[8] = {dispatch.u_data, dispatch.v_data, dispatch.w_data, dispatch.theta_data,
                                 dispatch.qv_data, dispatch.dz_prof, dispatch.dzc_prof, dispatch.rho};
        const void* outputs[8] = {dispatch.nu_t, dispatch.K_theta, dispatch.K_q, dispatch.dudt_sgs,
                                  dispatch.dvdt_sgs, dispatch.dwdt_sgs, dispatch.dthetadt_sgs,
                                  dispatch.dqvdt_sgs};
        const int first_output = dispatch.download_coefficients ? 0 : 3;
        for (int output = first_output; output < 8; ++output)
        {
            for (int prior = first_output; prior < output; ++prior)
                if (outputs[output] == outputs[prior])
                    return false;
            for (const void* input : inputs)
                if (outputs[output] == input)
                    return false;
        }

        id<MTLBuffer> u = [impl_->hardware.device newBufferWithBytes:dispatch.u_data length:field_bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> v = [impl_->hardware.device newBufferWithBytes:dispatch.v_data length:field_bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> w = [impl_->hardware.device newBufferWithBytes:dispatch.w_data length:field_bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> theta = [impl_->hardware.device newBufferWithBytes:dispatch.theta_data length:field_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> qv = [impl_->hardware.device newBufferWithBytes:dispatch.qv_data length:field_bytes
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> dz = [impl_->hardware.device newBufferWithBytes:dispatch.dz_prof length:profile_bytes
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> dzc = [impl_->hardware.device newBufferWithBytes:dispatch.dzc_prof length:profile_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> rho = [impl_->hardware.device newBufferWithBytes:dispatch.rho length:field_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> coefficients[3] = {
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
            [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared],
        };
        static const char* const tendency_names[5] = {
            "sgs_dudt", "sgs_dvdt", "sgs_dwdt", "sgs_dthetadt", "sgs_dqvdt"};
        const std::size_t tendency_sizes[5] = {
            field_bytes, field_bytes, field_bytes, field_bytes, field_bytes};
        Impl::ResidentAllocation tendency_allocations[5];
        if (!impl_->resident.fields.acquire_many(
                tendency_names, tendency_sizes, 5, tendency_allocations))
            return false;
        Impl::ResidentSlot* tendency_slots[5];
        id<MTLBuffer> tendencies[5];
        for (int index = 0; index < 5; ++index)
        {
            tendency_slots[index] = impl_->resident_slot(tendency_allocations[index]);
            if (tendency_slots[index] == nullptr || tendency_slots[index]->candidate == nil ||
                tendency_slots[index]->current == nil || tendency_slots[index]->bytes < field_bytes)
                return false;
            tendencies[index] = tendency_slots[index]->candidate;
        }
        if (u == nil || v == nil || w == nil || theta == nil || qv == nil || dz == nil || dzc == nil || rho == nil)
            return false;
        for (id<MTLBuffer> buffer : coefficients)
            if (buffer == nil)
                return false;
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> coefficient_encoder = [command_buffer computeCommandEncoder];
        if (coefficient_encoder == nil)
            return false;

        SmagorinskyConstants coefficient_constants{};
        coefficient_constants.nx = dispatch.nx;
        coefficient_constants.ny = dispatch.ny;
        coefficient_constants.nz = dispatch.nz;
        coefficient_constants.filter_mode = dispatch.filter_mode;
        coefficient_constants.use_horizontal = dispatch.use_horizontal ? 1 : 0;
        coefficient_constants.use_vertical = dispatch.use_vertical ? 1 : 0;
        coefficient_constants.dx = dispatch.dx_val;
        coefficient_constants.dy = dispatch.dy_val;
        coefficient_constants.Cs = dispatch.Cs;
        coefficient_constants.Pr_t = dispatch.Pr_t;
        coefficient_constants.Sc_t = dispatch.Sc_t;
        coefficient_constants.nu_t_max = dispatch.nu_t_max;
        coefficient_constants.K_max = dispatch.K_max;
        coefficient_constants.periodic_y = dispatch.periodic_y ? 1 : 0;
        coefficient_constants.periodic_x = dispatch.periodic_x ? 1 : 0;

        [coefficient_encoder setComputePipelineState:impl_->pipeline(KernelId::SmagorinskyCoefficients)];
        [coefficient_encoder setBuffer:u offset:0 atIndex:0];
        [coefficient_encoder setBuffer:v offset:0 atIndex:1];
        [coefficient_encoder setBuffer:w offset:0 atIndex:2];
        [coefficient_encoder setBuffer:dz offset:0 atIndex:3];
        [coefficient_encoder setBuffer:dzc offset:0 atIndex:4];
        for (NSUInteger index = 0; index < 3; ++index)
            [coefficient_encoder setBuffer:coefficients[index] offset:0 atIndex:index + 5];
        [coefficient_encoder setBytes:&coefficient_constants length:sizeof(coefficient_constants) atIndex:8];
        const NSUInteger coefficient_maximum =
            impl_->pipeline(KernelId::SmagorinskyCoefficients).maxTotalThreadsPerThreadgroup;
        if (coefficient_maximum == 0)
        {
            [coefficient_encoder endEncoding];
            return false;
        }
        [coefficient_encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, coefficient_maximum), 1, 1)];
        [coefficient_encoder endEncoding];

        id<MTLComputeCommandEncoder> tendency_encoder = [command_buffer computeCommandEncoder];
        if (tendency_encoder == nil)
            return false;
        SmagorinskyTendenciesConstants tendency_constants{};
        tendency_constants.nx = dispatch.nx;
        tendency_constants.ny = dispatch.ny;
        tendency_constants.nz = dispatch.nz;
        tendency_constants.use_horizontal = dispatch.use_horizontal ? 1 : 0;
        tendency_constants.use_vertical = dispatch.use_vertical ? 1 : 0;
        tendency_constants.has_qv = dispatch.has_qv ? 1 : 0;
        tendency_constants.periodic_x = dispatch.periodic_x ? 1 : 0;
        tendency_constants.periodic_y = dispatch.periodic_y ? 1 : 0;
        tendency_constants.dx = dispatch.dx_val;
        tendency_constants.dy = dispatch.dy_val;
        tendency_constants.closed_lateral = turbulence_closed_lateral_mask(dispatch.open_lateral);

        id<MTLBuffer> tendency_inputs[10] = {u, v, w, theta, qv, coefficients[0], coefficients[1],
                                             coefficients[2], dz, rho};
        [tendency_encoder setComputePipelineState:impl_->pipeline(KernelId::SmagorinskyTendencies)];
        for (NSUInteger index = 0; index < 10; ++index)
            [tendency_encoder setBuffer:tendency_inputs[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 5; ++index)
            [tendency_encoder setBuffer:tendencies[index] offset:0 atIndex:index + 10];
        [tendency_encoder setBytes:&tendency_constants length:sizeof(tendency_constants) atIndex:15];
        const NSUInteger tendency_maximum = impl_->pipeline(KernelId::SmagorinskyTendencies).maxTotalThreadsPerThreadgroup;
        if (tendency_maximum == 0)
        {
            [tendency_encoder endEncoding];
            return false;
        }
        [tendency_encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, tendency_maximum), 1, 1)];
        [tendency_encoder endEncoding];

        if (!impl_->finish(command_buffer))
            return false;

        for (int index = 0; index < 5; ++index)
            std::swap(tendency_slots[index]->current, tendency_slots[index]->candidate);

        if (dispatch.download_coefficients)
        {
            void* coefficient_destinations[3] = {dispatch.nu_t, dispatch.K_theta, dispatch.K_q};
            for (int index = 0; index < 3; ++index)
                std::memcpy(coefficient_destinations[index], coefficients[index].contents, field_bytes);
        }
        void* tendency_destinations[5] = {dispatch.dudt_sgs, dispatch.dvdt_sgs, dispatch.dwdt_sgs,
                                          dispatch.dthetadt_sgs, dispatch.dqvdt_sgs};
        for (int index = 0; index < 5; ++index)
        {
            std::memcpy(tendency_destinations[index], tendency_slots[index]->current.contents, field_bytes);
            const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
            impl_->resident.fields.note_device_write(tendency_names[index], tendency_destinations[index], extent);
            impl_->resident.fields.note_host_synced(tendency_names[index]);
        }

        ++impl_->telemetry.completed_smagorinsky_coefficients_dispatches;
        ++impl_->telemetry.completed_smagorinsky_tendencies_dispatches;
        ++impl_->telemetry.completed_smagorinsky_fused_dispatches;
        return true;
    }
}

bool MetalComputeBackend::supports_sgs_apply() const noexcept
{
    return ready(nimbus::compute::KernelId::SgsApply);
}

bool MetalComputeBackend::dispatch_sgs_apply(const SgsApplyDispatch& dispatch)
{
    @autoreleasepool
    {
        if (dispatch.collocated && (dispatch.nx < 3 || dispatch.ny < 3 || dispatch.nz < 3))
            return false;
        std::size_t cells       = 0;
        std::size_t field_bytes = 0;
        if (!supports_sgs_apply() || dispatch.u == nullptr || dispatch.v == nullptr || dispatch.w == nullptr ||
            dispatch.theta_data == nullptr || dispatch.qv_data == nullptr || dispatch.tke_data == nullptr ||
            dispatch.dudt == nullptr || dispatch.dvdt == nullptr || dispatch.dwdt == nullptr ||
            dispatch.dthetadt == nullptr || dispatch.dqvdt == nullptr || dispatch.dtkedt == nullptr ||
            dispatch.dtkedt_pbl == nullptr || !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            !std::isfinite(dispatch.dt_step) || dispatch.dt_step <= 0.0f ||
            !std::isfinite(dispatch.theta_fallback) || !std::isfinite(dispatch.theta_min) ||
            !std::isfinite(dispatch.theta_max) || dispatch.theta_min > dispatch.theta_max ||
            !std::isfinite(dispatch.qv_max_val) || dispatch.qv_max_val < 0.0f ||
            !std::isfinite(dispatch.wind_h_max) || dispatch.wind_h_max < 0.0f ||
            !std::isfinite(dispatch.wind_v_max) || dispatch.wind_v_max < 0.0f ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        void* state_fields[6] = {dispatch.u, dispatch.v, dispatch.w, dispatch.theta_data,
                                 dispatch.qv_data, dispatch.tke_data};
        const void* tendency_fields[7] = {dispatch.dudt, dispatch.dvdt, dispatch.dwdt, dispatch.dthetadt,
                                          dispatch.dqvdt, dispatch.dtkedt, dispatch.dtkedt_pbl};
        for (int field = 0; field < 6; ++field)
        {
            for (int prior = 0; prior < field; ++prior)
                if (state_fields[field] == state_fields[prior])
                    return false;
            for (const void* tendency : tendency_fields)
                if (state_fields[field] == tendency)
                    return false;
        }
        for (int field = 0; field < 7; ++field)
            for (int prior = 0; prior < field; ++prior)
                if (tendency_fields[field] == tendency_fields[prior])
                    return false;

        id<MTLBuffer> inputs[13] = {};
        const void* input_data[13] = {dispatch.u, dispatch.v, dispatch.w, dispatch.theta_data, dispatch.qv_data,
                                      dispatch.tke_data, dispatch.dudt, dispatch.dvdt, dispatch.dwdt,
                                      dispatch.dthetadt, dispatch.dqvdt, dispatch.dtkedt, dispatch.dtkedt_pbl};
        for (int index = 0; index < 6; ++index)
        {
            inputs[index] = [impl_->hardware.device newBufferWithBytes:input_data[index]
                                                        length:field_bytes
                                                       options:MTLResourceStorageModeShared];
            if (inputs[index] == nil)
                return false;
        }
        static const char* const tendency_names[5] = {
            "sgs_dudt", "sgs_dvdt", "sgs_dwdt", "sgs_dthetadt", "sgs_dqvdt"};
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        bool use_resident_tendencies = dispatch.resident_tendencies_unchanged;
        Impl::ResidentSlot* tendency_slots[5] = {};
        if (use_resident_tendencies)
        {
            for (int index = 0; index < 5; ++index)
            {
                const bool reusable = impl_->resident.fields.reusable(
                    tendency_names[index], tendency_fields[index], extent, true);
                const auto* entry = impl_->resident.fields.find(tendency_names[index]);
                tendency_slots[index] = entry == nullptr ? nullptr : impl_->resident_slot(entry->allocation);
                if (!reusable || tendency_slots[index] == nullptr || tendency_slots[index]->current == nil ||
                    tendency_slots[index]->bytes < field_bytes)
                {
                    use_resident_tendencies = false;
                    break;
                }
            }
        }
        for (int index = 0; index < 5; ++index)
        {
            if (use_resident_tendencies)
            {
                inputs[index + 6] = tendency_slots[index]->current;
            }
            else
            {
                inputs[index + 6] = [impl_->hardware.device newBufferWithBytes:input_data[index + 6]
                                                                length:field_bytes
                                                               options:MTLResourceStorageModeShared];
                if (inputs[index + 6] == nil)
                    return false;
            }
        }
        for (int index = 11; index < 13; ++index)
        {
            inputs[index] = [impl_->hardware.device newBufferWithBytes:input_data[index]
                                                        length:field_bytes
                                                       options:MTLResourceStorageModeShared];
            if (inputs[index] == nil)
                return false;
        }
        id<MTLBuffer> outputs[6];
        for (int index = 0; index < 6; ++index)
        {
            outputs[index] = [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
            if (outputs[index] == nil)
                return false;
        }

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;

        SgsApplyConstants constants{};
        constants.nx             = dispatch.nx;
        constants.ny             = dispatch.ny;
        constants.nz             = dispatch.nz;
        constants.has_qv         = dispatch.has_qv ? 1 : 0;
        constants.has_tke        = dispatch.has_tke ? 1 : 0;
        constants.dt_step        = dispatch.dt_step;
        constants.theta_fallback = dispatch.theta_fallback;
        constants.theta_min      = dispatch.theta_min;
        constants.theta_max      = dispatch.theta_max;
        constants.qv_max         = dispatch.qv_max_val;
        constants.wind_h_max     = dispatch.wind_h_max;
        constants.wind_v_max     = dispatch.wind_v_max;
        constants.sanitize       = dispatch.sanitize ? 1 : 0;
        constants.collocated     = dispatch.collocated ? 1 : 0;

        [encoder setComputePipelineState:impl_->pipeline(KernelId::SgsApply)];
        for (NSUInteger index = 0; index < 13; ++index)
            [encoder setBuffer:inputs[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 6; ++index)
            [encoder setBuffer:outputs[index] offset:0 atIndex:index + 13];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:19];
        const NSUInteger maximum = impl_->pipeline(KernelId::SgsApply).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        ++impl_->telemetry.completed_sgs_apply_dispatches;
        if (use_resident_tendencies)
            impl_->telemetry.sgs_resident_tendency_reuses += 5;
        if (dispatch.completed_observer != nullptr)
        {
            dispatch.completed_observer(dispatch.completed_observer_context,
                                        static_cast<const float*>(outputs[0].contents),
                                        static_cast<const float*>(outputs[1].contents),
                                        static_cast<const float*>(outputs[2].contents),
                                        static_cast<const float*>(outputs[3].contents),
                                        static_cast<const float*>(outputs[4].contents),
                                        static_cast<const float*>(outputs[5].contents));
        }
        for (int index = 0; index < 6; ++index)
            std::memcpy(state_fields[index], outputs[index].contents, field_bytes);
        return true;
    }
}

namespace
{
template <class Implementation, class Constants, std::size_t N>
bool dispatch_tke_fields_metal(Implementation&                    impl,
                               KernelId                           kernel,
                               const Constants&                   constants,
                               const std::array<const float*, 7>& inputs,
                               const std::array<float*, N>&       outputs,
                               std::size_t                        cells,
                               int                                nz)
{
    const std::size_t bytes = cells * sizeof(float);
    if (bytes > impl.max_buffer_bytes())
        return false;
    id<MTLBuffer> buffers[7 + N];
    for (std::size_t i = 0; i < 7; ++i)
        buffers[i] =
            [impl.hardware.device newBufferWithBytes:inputs[i]
                                              length:(i == 6 ? static_cast<std::size_t>(nz) * sizeof(float) : bytes)
                                             options:MTLResourceStorageModeShared];
    for (std::size_t i = 0; i < N; ++i)
        buffers[7 + i] = [impl.hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    for (auto buffer : buffers)
        if (buffer == nil)
            return false;
    auto pipeline = impl.pipeline(kernel);
    if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0)
        return false;
    id<MTLCommandBuffer> command = impl.command_buffer();
    if (command == nil)
        return false;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil)
        return false;
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < 7 + N; ++i)
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder setBytes:&constants length:sizeof(constants) atIndex:7 + N];
    [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    if (!impl.finish(command))
        return false;
    if (kernel == KernelId::TkeClosure)
        ++impl.telemetry.completed_tke_closure_dispatches;
    else
        ++impl.telemetry.completed_tke_tendency_dispatches;
    for (std::size_t f = 0; f < N; ++f)
    {
        auto data = static_cast<const float*>(buffers[7 + f].contents);
        if (data == nullptr)
            return false;
        for (std::size_t n = 0; n < cells; ++n)
            if (!std::isfinite(data[n]) || (kernel == KernelId::TkeClosure && f != 5 && data[n] < 0))
                return false;
    }
    for (std::size_t f = 0; f < N; ++f)
        std::memcpy(outputs[f], buffers[7 + f].contents, bytes);
    return true;
}
} // namespace

bool MetalComputeBackend::dispatch_tke_closure(const TkeClosureDispatch& d)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        if (!supports(TurbulenceKernelOp::Tke) || !nimbus::compute::tke_contract::validate(d, cells))
            return false;
        return dispatch_tke_fields_metal(
            *impl_,
            KernelId::TkeClosure,
            nimbus::compute::tke_contract::constants(d),
            std::array<const float*, 7>{d.u, d.v, d.w, d.theta, d.qv, d.tke, d.dz},
            std::array<float*, 8>{
                d.nu, d.K_theta, d.K_q, d.K_tke, d.shear, d.buoyancy, d.dissipation, d.effective_prandtl},
            cells,
            d.nz);
    }
}

bool MetalComputeBackend::dispatch_tke_tendency(const TkeTendencyDispatch& d)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        if (!supports(TurbulenceKernelOp::Tke) || !nimbus::compute::tke_contract::validate(d, cells))
            return false;
        return dispatch_tke_fields_metal(
            *impl_,
            KernelId::TkeTendency,
            nimbus::compute::tke_contract::constants(d),
            std::array<const float*, 7>{d.rho, d.tke, d.K_tke, d.shear, d.buoyancy, d.dissipation, d.dz},
            std::array<float*, 1>{d.tendency},
            cells,
            d.nz);
    }
}
