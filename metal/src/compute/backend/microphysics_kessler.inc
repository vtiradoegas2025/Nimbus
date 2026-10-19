// Microphysics routing and Kessler pointwise/sedimentation/fused execution.
MicrophysicsComputeKernels* MetalComputeBackend::microphysics_kernels()
{
    return supports_kessler_pointwise() && supports_kessler_sedimentation() ? this : nullptr;
}

const MicrophysicsComputeKernels* MetalComputeBackend::microphysics_kernels() const
{
    return supports_kessler_pointwise() && supports_kessler_sedimentation() ? this : nullptr;
}

bool MetalComputeBackend::supports(MicrophysicsKernelOp op) const
{
    if (op == MicrophysicsKernelOp::ThompsonCompletion)
    {
        return ready(nimbus::compute::KernelId::ThompsonCompletion);
    }
    if (op == MicrophysicsKernelOp::ThompsonDeposition)
    {
        return ready(nimbus::compute::KernelId::ThompsonDeposition);
    }
    if (op == MicrophysicsKernelOp::ThompsonFreezing)
    {
        return ready(nimbus::compute::KernelId::ThompsonFreezing);
    }
    if (op == MicrophysicsKernelOp::ThompsonCloudPhase)
    {
        return ready(nimbus::compute::KernelId::ThompsonCloudPhase);
    }
    if (op == MicrophysicsKernelOp::MilbrandtMelting)
        return ready(nimbus::compute::KernelId::MilbrandtMelting);
    if (op == MicrophysicsKernelOp::MilbrandtSedimentation)
        return ready(nimbus::compute::KernelId::MilbrandtSedimentation);
    if (op == MicrophysicsKernelOp::MilbrandtCold)
        return ready(nimbus::compute::KernelId::MilbrandtCold);
    if (op == MicrophysicsKernelOp::MilbrandtCloudAdjustment)
        return ready(nimbus::compute::KernelId::MilbrandtCloudAdjustment);
    if (op == MicrophysicsKernelOp::MilbrandtWarmRain)
        return ready(nimbus::compute::KernelId::MilbrandtWarmRain);
    if (op == MicrophysicsKernelOp::ThompsonWarmRain)
    {
        return ready(nimbus::compute::KernelId::ThompsonWarmRain);
    }
    if (op == MicrophysicsKernelOp::ThompsonCloudAdjustment)
    {
        return ready(nimbus::compute::KernelId::ThompsonCloudAdjustment);
    }
    if (op == MicrophysicsKernelOp::ThompsonMomentSedimentation)
    {
        return ready(nimbus::compute::KernelId::ThompsonMomentSedimentation);
    }
    if (op == MicrophysicsKernelOp::LinFused)
        return supports_lin_pointwise() && supports_lin_sedimentation();
    if (op == MicrophysicsKernelOp::LinPointwise)
        return supports_lin_pointwise();
    if (op == MicrophysicsKernelOp::LinSedimentation)
        return supports_lin_sedimentation();
    if (op == MicrophysicsKernelOp::KesslerPointwise)
        return supports_kessler_pointwise();
    if (op == MicrophysicsKernelOp::KesslerSedimentation)
        return supports_kessler_sedimentation();
    if (op == MicrophysicsKernelOp::MicrophysicsApply)
        return supports_microphysics_apply();
    return false;
}

bool MetalComputeBackend::supports_kessler_pointwise() const noexcept
{
    return ready(nimbus::compute::KernelId::KesslerPointwise);
}

bool MetalComputeBackend::supports_kessler_sedimentation() const noexcept
{
    return ready(nimbus::compute::KernelId::KesslerSedimentation);
}

bool MetalComputeBackend::dispatch_kessler_pointwise(const KesslerPointwiseDispatch& dispatch)
{
    @autoreleasepool
    {
        const float* inputs_data[7] = {dispatch.temperature_data, dispatch.p_data, dispatch.qv_data,
                                       dispatch.qc_data, dispatch.qr_data, dispatch.rho_d_data,
                                       dispatch.qh_data};
        float* outputs_data[6] = {dispatch.dtheta_dt_data, dispatch.dqv_dt_data, dispatch.dqc_dt_data,
                                  dispatch.dqr_dt_data, dispatch.dqg_dt_data, dispatch.dqh_dt_data};
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        if (!supports_kessler_pointwise() ||
            dispatch.i_begin < 0 || dispatch.i_begin >= dispatch.i_end || dispatch.i_end > dispatch.nr ||
            dispatch.j_begin < 0 || dispatch.j_begin >= dispatch.j_end || dispatch.j_end > dispatch.nth ||
            dispatch.k_begin < 0 || dispatch.k_begin >= dispatch.k_end || dispatch.k_end > dispatch.nz ||
            !std::isfinite(dispatch.qc0) || dispatch.qc0 < 0.0f ||
            !std::isfinite(dispatch.c_auto) || dispatch.c_auto < 0.0f ||
            !std::isfinite(dispatch.c_accr) || dispatch.c_accr < 0.0f ||
            !std::isfinite(dispatch.evaporation_efficiency) || dispatch.evaporation_efficiency < 0.0f ||
            !std::isfinite(dispatch.Lv_cp) || dispatch.Lv_cp <= 0.0f ||
            !std::isfinite(dispatch.T0) || dispatch.T0 <= 0.0f ||
            !std::isfinite(dispatch.dt_step) || dispatch.dt_step <= 0.0f ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        for (const float* input : inputs_data)
            if (input == nullptr)
                return false;
        for (float* output : outputs_data)
            if (output == nullptr)
                return false;
        for (int output = 0; output < 6; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs_data[output] == outputs_data[prior])
                    return false;
            for (const float* input : inputs_data)
                if (outputs_data[output] == input)
                    return false;
        }
        for (int i = dispatch.i_begin; i < dispatch.i_end; ++i)
            for (int j = dispatch.j_begin; j < dispatch.j_end; ++j)
                for (int k = dispatch.k_begin; k < dispatch.k_end; ++k)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(i) * static_cast<std::size_t>(dispatch.nth) +
                         static_cast<std::size_t>(j)) * static_cast<std::size_t>(dispatch.nz) +
                        static_cast<std::size_t>(k);
                    for (const float* input : inputs_data)
                        if (!std::isfinite(input[index]))
                            return false;
                    if (dispatch.p_data[index] <= 0.0f ||
                        dispatch.rho_d_data[index] <= 0.0f)
                        return false;
                }

        id<MTLBuffer> inputs[7] = {};
        id<MTLBuffer> outputs[6] = {};
        for (int index = 0; index < 7; ++index)
        {
            inputs[index] = impl_->kessler_buffer(index, field_bytes, inputs_data[index]);
            if (inputs[index] == nil)
                return false;
        }
        for (int index = 0; index < 6; ++index)
        {
            outputs[index] = impl_->kessler_buffer(index + 7, field_bytes);
            if (outputs[index] == nil)
                return false;
        }

        KesslerPointwiseConstants constants{};
        constants.nr = dispatch.nr;
        constants.nth = dispatch.nth;
        constants.nz = dispatch.nz;
        constants.i_begin = dispatch.i_begin;
        constants.i_end = dispatch.i_end;
        constants.j_begin = dispatch.j_begin;
        constants.j_end = dispatch.j_end;
        constants.k_begin = dispatch.k_begin;
        constants.k_end = dispatch.k_end;
        constants.qc0 = dispatch.qc0;
        constants.c_auto = dispatch.c_auto;
        constants.c_accr = dispatch.c_accr;
        constants.evaporation_efficiency = dispatch.evaporation_efficiency;
        constants.Lv_cp = dispatch.Lv_cp;
        constants.T0 = dispatch.T0;
        constants.dt_step = dispatch.dt_step;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::KesslerPointwise).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::KesslerPointwise)];
        for (NSUInteger index = 0; index < 7; ++index)
            [encoder setBuffer:inputs[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 6; ++index)
            [encoder setBuffer:outputs[index] offset:0 atIndex:index + 7];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:13];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        for (int index = 0; index < 6; ++index)
            std::memcpy(outputs_data[index], outputs[index].contents, field_bytes);
        ++impl_->telemetry.completed_kessler_pointwise_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_kessler_sedimentation(const KesslerSedimentationDispatch& dispatch)
{
    @autoreleasepool
    {
        const float* inputs_data[5]  = {dispatch.qr_data,
                                        dispatch.qg_data,
                                        dispatch.qh_data,
                                        dispatch.rho_d_data,
                                        dispatch.layer_thickness_m ? dispatch.layer_thickness_m : dispatch.rho_d_data};
        float*       outputs_data[3] = {dispatch.dqr_dt_data, dispatch.dqg_dt_data, dispatch.dqh_dt_data};
        std::size_t  cells           = 0;
        std::size_t  field_bytes     = 0;
        std::size_t  columns         = 0;
        std::size_t  ignored_bytes   = 0;
        if (!supports_kessler_sedimentation() || !nimbus::compute::valid_kessler_sedimentation(dispatch) ||
            dispatch.i_begin < 0 || dispatch.i_begin >= dispatch.i_end || dispatch.i_end > dispatch.nr ||
            dispatch.j_begin < 0 || dispatch.j_begin >= dispatch.j_end || dispatch.j_end > dispatch.nth ||
            dispatch.k_begin < 0 || dispatch.k_begin >= dispatch.k_end || dispatch.k_end > dispatch.nz ||
            !std::isfinite(dispatch.dz_val) || dispatch.dz_val <= 0.0f || !std::isfinite(dispatch.Vt_max_rain) ||
            dispatch.Vt_max_rain < 0.0f || !std::isfinite(dispatch.vt_cap) || dispatch.vt_cap < 0.0f ||
            !std::isfinite(dispatch.dt_step) || dispatch.dt_step <= 0.0f ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            !checked_bytes(dispatch.nr, dispatch.nth, 1, columns, ignored_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        for (const float* input : inputs_data)
            if (input == nullptr)
                return false;
        for (float* output : outputs_data)
            if (output == nullptr)
                return false;
        for (int output = 0; output < 3; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs_data[output] == outputs_data[prior])
                    return false;
            for (const float* input : inputs_data)
                if (outputs_data[output] == input)
                    return false;
        }
        for (int i = dispatch.i_begin; i < dispatch.i_end; ++i)
            for (int j = dispatch.j_begin; j < dispatch.j_end; ++j)
                for (int k = dispatch.k_begin; k < dispatch.k_end; ++k)
                {
                    const std::size_t index = (static_cast<std::size_t>(i) * static_cast<std::size_t>(dispatch.nth) +
                                               static_cast<std::size_t>(j)) *
                                                  static_cast<std::size_t>(dispatch.nz) +
                                              static_cast<std::size_t>(k);
                    for (const float* input : inputs_data)
                        if (!std::isfinite(input[index]))
                            return false;
                    for (float* output : outputs_data)
                        if (!std::isfinite(output[index]))
                            return false;
                    if (dispatch.rho_d_data[index] <= 0.0f)
                        return false;
                }

        id<MTLBuffer> inputs[5]  = {};
        id<MTLBuffer> outputs[3] = {};
        for (int index = 0; index < 5; ++index)
        {
            inputs[index] = impl_->kessler_buffer(index, field_bytes, inputs_data[index]);
            if (inputs[index] == nil)
                return false;
        }
        for (int index = 0; index < 3; ++index)
        {
            outputs[index] = impl_->kessler_buffer(index + 7, field_bytes, outputs_data[index]);
            if (outputs[index] == nil)
                return false;
        }

        KesslerSedimentationConstants constants{};
        constants.nr                   = dispatch.nr;
        constants.nth                  = dispatch.nth;
        constants.nz                   = dispatch.nz;
        constants.i_begin              = dispatch.i_begin;
        constants.i_end                = dispatch.i_end;
        constants.j_begin              = dispatch.j_begin;
        constants.j_end                = dispatch.j_end;
        constants.k_begin              = dispatch.k_begin;
        constants.k_end                = dispatch.k_end;
        constants.dz_val               = dispatch.dz_val;
        constants.Vt_max_rain          = dispatch.Vt_max_rain;
        constants.vt_cap               = dispatch.vt_cap;
        constants.have_rho_d           = 1;
        constants.dt_step              = dispatch.dt_step;
        constants.have_layer_thickness = dispatch.layer_thickness_m != nullptr;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::KesslerSedimentation).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::KesslerSedimentation)];
        for (NSUInteger index = 0; index < 5; ++index)
            [encoder setBuffer:inputs[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 3; ++index)
            [encoder setBuffer:outputs[index] offset:0 atIndex:index + 5];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:8];
        [encoder dispatchThreads:MTLSizeMake(columns, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        if (!nimbus::compute::finite_kessler_sedimentation(dispatch,
                                                           {static_cast<const float*>(outputs[0].contents),
                                                            static_cast<const float*>(outputs[1].contents),
                                                            static_cast<const float*>(outputs[2].contents)}))
            return false;
        for (int index = 0; index < 3; ++index)
            std::memcpy(outputs_data[index], outputs[index].contents, field_bytes);
        ++impl_->telemetry.completed_kessler_sedimentation_dispatches;
        return true;
    }
}

bool MetalComputeBackend::supports_kessler_fused() const noexcept
{
    return supports_kessler_pointwise() && supports_kessler_sedimentation();
}

bool MetalComputeBackend::dispatch_kessler_fused(const MetalKesslerFusedDispatch& dispatch)
{
    @autoreleasepool
    {
        const KesslerPointwiseDispatch& pointwise = dispatch.pointwise;
        const KesslerSedimentationDispatch& sedimentation = dispatch.sedimentation;
        const float* inputs_data[8] = {
            pointwise.temperature_data, pointwise.p_data, pointwise.qv_data, pointwise.qc_data,
            pointwise.qr_data, pointwise.rho_d_data, pointwise.qh_data, sedimentation.qg_data};
        float* outputs_data[6] = {
            pointwise.dtheta_dt_data, pointwise.dqv_dt_data, pointwise.dqc_dt_data,
            pointwise.dqr_dt_data, pointwise.dqg_dt_data, pointwise.dqh_dt_data};
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        std::size_t columns = 0;
        std::size_t ignored_bytes = 0;
        if (!supports_kessler_fused() || sedimentation.layer_thickness_m != nullptr ||
            pointwise.i_begin < 0 || pointwise.i_begin >= pointwise.i_end || pointwise.i_end > pointwise.nr ||
            pointwise.j_begin < 0 || pointwise.j_begin >= pointwise.j_end || pointwise.j_end > pointwise.nth ||
            pointwise.k_begin < 0 || pointwise.k_begin >= pointwise.k_end || pointwise.k_end > pointwise.nz ||
            !std::isfinite(pointwise.qc0) || pointwise.qc0 < 0.0f ||
            !std::isfinite(pointwise.c_auto) || pointwise.c_auto < 0.0f ||
            !std::isfinite(pointwise.c_accr) || pointwise.c_accr < 0.0f ||
            !std::isfinite(pointwise.evaporation_efficiency) || pointwise.evaporation_efficiency < 0.0f ||
            !std::isfinite(pointwise.Lv_cp) || pointwise.Lv_cp <= 0.0f ||
            !std::isfinite(pointwise.T0) || pointwise.T0 <= 0.0f ||
            !std::isfinite(pointwise.dt_step) || pointwise.dt_step <= 0.0f ||
            !std::isfinite(sedimentation.dz_val) || sedimentation.dz_val <= 0.0f ||
            !std::isfinite(sedimentation.Vt_max_rain) || sedimentation.Vt_max_rain < 0.0f ||
            !std::isfinite(sedimentation.vt_cap) || sedimentation.vt_cap < 0.0f ||
            !std::isfinite(sedimentation.dt_step) || sedimentation.dt_step <= 0.0f ||
            pointwise.nr != sedimentation.nr || pointwise.nth != sedimentation.nth ||
            pointwise.nz != sedimentation.nz || pointwise.i_begin != sedimentation.i_begin ||
            pointwise.i_end != sedimentation.i_end || pointwise.j_begin != sedimentation.j_begin ||
            pointwise.j_end != sedimentation.j_end || pointwise.k_begin != sedimentation.k_begin ||
            pointwise.k_end != sedimentation.k_end || pointwise.dt_step != sedimentation.dt_step ||
            pointwise.qr_data != sedimentation.qr_data || pointwise.rho_d_data != sedimentation.rho_d_data ||
            pointwise.qh_data != sedimentation.qh_data ||
            pointwise.dqr_dt_data != sedimentation.dqr_dt_data ||
            pointwise.dqg_dt_data != sedimentation.dqg_dt_data ||
            pointwise.dqh_dt_data != sedimentation.dqh_dt_data ||
            !checked_bytes(pointwise.nr, pointwise.nth, pointwise.nz, cells, field_bytes) ||
            !checked_bytes(pointwise.nr, pointwise.nth, 1, columns, ignored_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        nimbus::compute::DispatchGraph schedule;
        const auto pointwise_node = schedule.add(nimbus::compute::KernelId::KesslerPointwise, {
            {"temperature", nimbus::compute::ResourceAccess::Read},
            {"pressure", nimbus::compute::ResourceAccess::Read},
            {"qv", nimbus::compute::ResourceAccess::Read},
            {"qc", nimbus::compute::ResourceAccess::Read},
            {"qr", nimbus::compute::ResourceAccess::Read},
            {"rho_d", nimbus::compute::ResourceAccess::Read},
            {"qh", nimbus::compute::ResourceAccess::Read},
            {"dtheta_dt", nimbus::compute::ResourceAccess::Write},
            {"dqv_dt", nimbus::compute::ResourceAccess::Write},
            {"dqc_dt", nimbus::compute::ResourceAccess::Write},
            {"dqr_dt", nimbus::compute::ResourceAccess::Write},
            {"dqg_dt", nimbus::compute::ResourceAccess::Write},
            {"dqh_dt", nimbus::compute::ResourceAccess::Write}});
        const auto sedimentation_node = schedule.add(nimbus::compute::KernelId::KesslerSedimentation, {
            {"qr", nimbus::compute::ResourceAccess::Read},
            {"qg", nimbus::compute::ResourceAccess::Read},
            {"qh", nimbus::compute::ResourceAccess::Read},
            {"rho_d", nimbus::compute::ResourceAccess::Read},
            {"dqr_dt", nimbus::compute::ResourceAccess::ReadWrite},
            {"dqg_dt", nimbus::compute::ResourceAccess::ReadWrite},
            {"dqh_dt", nimbus::compute::ResourceAccess::ReadWrite}});
        // Separate Metal encoders form the required visibility boundary. Do
        // not encode the bound stages if the canonical hazard schedule does
        // not require that ordering.
        if (!schedule.valid() || !schedule.depends_on(sedimentation_node, pointwise_node) ||
            !schedule.requires_barrier(pointwise_node, sedimentation_node))
            return false;
        for (const float* input : inputs_data)
            if (input == nullptr)
                return false;
        for (float* output : outputs_data)
            if (output == nullptr)
                return false;
        for (int output = 0; output < 6; ++output)
        {
            for (int prior = 0; prior < output; ++prior)
                if (outputs_data[output] == outputs_data[prior])
                    return false;
            for (const float* input : inputs_data)
                if (outputs_data[output] == input)
                    return false;
        }
        for (int i = pointwise.i_begin; i < pointwise.i_end; ++i)
            for (int j = pointwise.j_begin; j < pointwise.j_end; ++j)
                for (int k = pointwise.k_begin; k < pointwise.k_end; ++k)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(i) * static_cast<std::size_t>(pointwise.nth) +
                         static_cast<std::size_t>(j)) * static_cast<std::size_t>(pointwise.nz) +
                        static_cast<std::size_t>(k);
                    for (const float* input : inputs_data)
                        if (!std::isfinite(input[index]))
                            return false;
                    if (pointwise.p_data[index] <= 0.0f || pointwise.rho_d_data[index] <= 0.0f)
                        return false;
                }

        id<MTLBuffer> inputs[8] = {};
        for (int index = 0; index < 8; ++index)
        {
            inputs[index] = impl_->kessler_buffer(index, field_bytes, inputs_data[index]);
            if (inputs[index] == nil)
                return false;
        }

        static const char* const tendency_names[6] = {
            "kessler_dtheta_dt", "kessler_dqv_dt", "kessler_dqc_dt",
            "kessler_dqr_dt", "kessler_dqg_dt", "kessler_dqh_dt"};
        const std::size_t tendency_sizes[6] = {
            field_bytes, field_bytes, field_bytes, field_bytes, field_bytes, field_bytes};
        Impl::ResidentAllocation tendency_allocations[6];
        if (!impl_->resident.fields.acquire_many(
                tendency_names, tendency_sizes, 6, tendency_allocations))
            return false;
        Impl::ResidentSlot* tendency_slots[6] = {};
        id<MTLBuffer> tendencies[6] = {};
        for (int index = 0; index < 6; ++index)
        {
            tendency_slots[index] = impl_->resident_slot(tendency_allocations[index]);
            if (tendency_slots[index] == nullptr || tendency_slots[index]->candidate == nil ||
                tendency_slots[index]->current == nil || tendency_slots[index]->bytes < field_bytes)
                return false;
            tendencies[index] = tendency_slots[index]->candidate;
        }

        KesslerPointwiseConstants pointwise_constants{};
        pointwise_constants.nr = pointwise.nr;
        pointwise_constants.nth = pointwise.nth;
        pointwise_constants.nz = pointwise.nz;
        pointwise_constants.i_begin = pointwise.i_begin;
        pointwise_constants.i_end = pointwise.i_end;
        pointwise_constants.j_begin = pointwise.j_begin;
        pointwise_constants.j_end = pointwise.j_end;
        pointwise_constants.k_begin = pointwise.k_begin;
        pointwise_constants.k_end = pointwise.k_end;
        pointwise_constants.qc0 = pointwise.qc0;
        pointwise_constants.c_auto = pointwise.c_auto;
        pointwise_constants.c_accr = pointwise.c_accr;
        pointwise_constants.evaporation_efficiency = pointwise.evaporation_efficiency;
        pointwise_constants.Lv_cp = pointwise.Lv_cp;
        pointwise_constants.T0 = pointwise.T0;
        pointwise_constants.dt_step = pointwise.dt_step;

        KesslerSedimentationConstants sedimentation_constants{};
        sedimentation_constants.nr = sedimentation.nr;
        sedimentation_constants.nth = sedimentation.nth;
        sedimentation_constants.nz = sedimentation.nz;
        sedimentation_constants.i_begin = sedimentation.i_begin;
        sedimentation_constants.i_end = sedimentation.i_end;
        sedimentation_constants.j_begin = sedimentation.j_begin;
        sedimentation_constants.j_end = sedimentation.j_end;
        sedimentation_constants.k_begin = sedimentation.k_begin;
        sedimentation_constants.k_end = sedimentation.k_end;
        sedimentation_constants.dz_val = sedimentation.dz_val;
        sedimentation_constants.Vt_max_rain = sedimentation.Vt_max_rain;
        sedimentation_constants.vt_cap = sedimentation.vt_cap;
        sedimentation_constants.have_rho_d = 1;
        sedimentation_constants.dt_step = sedimentation.dt_step;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> pointwise_encoder = [command_buffer computeCommandEncoder];
        if (pointwise_encoder == nil)
            return false;
        const NSUInteger pointwise_maximum = impl_->pipeline(KernelId::KesslerPointwise).maxTotalThreadsPerThreadgroup;
        if (pointwise_maximum == 0)
        {
            [pointwise_encoder endEncoding];
            return false;
        }
        [pointwise_encoder setComputePipelineState:impl_->pipeline(KernelId::KesslerPointwise)];
        for (NSUInteger index = 0; index < 7; ++index)
            [pointwise_encoder setBuffer:inputs[index] offset:0 atIndex:index];
        for (NSUInteger index = 0; index < 6; ++index)
            [pointwise_encoder setBuffer:tendencies[index] offset:0 atIndex:index + 7];
        [pointwise_encoder setBytes:&pointwise_constants length:sizeof(pointwise_constants) atIndex:13];
        [pointwise_encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pointwise_maximum), 1, 1)];
        [pointwise_encoder endEncoding];

        id<MTLComputeCommandEncoder> sedimentation_encoder = [command_buffer computeCommandEncoder];
        if (sedimentation_encoder == nil)
            return false;
        const NSUInteger sedimentation_maximum =
            impl_->pipeline(KernelId::KesslerSedimentation).maxTotalThreadsPerThreadgroup;
        if (sedimentation_maximum == 0)
        {
            [sedimentation_encoder endEncoding];
            return false;
        }
        [sedimentation_encoder setComputePipelineState:impl_->pipeline(KernelId::KesslerSedimentation)];
        [sedimentation_encoder setBuffer:inputs[4] offset:0 atIndex:0];
        [sedimentation_encoder setBuffer:inputs[7] offset:0 atIndex:1];
        [sedimentation_encoder setBuffer:inputs[6] offset:0 atIndex:2];
        [sedimentation_encoder setBuffer:inputs[5] offset:0 atIndex:3];
        [sedimentation_encoder setBuffer:inputs[5] offset:0 atIndex:4];
        [sedimentation_encoder setBuffer:tendencies[3] offset:0 atIndex:5];
        [sedimentation_encoder setBuffer:tendencies[4] offset:0 atIndex:6];
        [sedimentation_encoder setBuffer:tendencies[5] offset:0 atIndex:7];
        [sedimentation_encoder setBytes:&sedimentation_constants
                                  length:sizeof(sedimentation_constants)
                                 atIndex:8];
        [sedimentation_encoder dispatchThreads:MTLSizeMake(columns, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, sedimentation_maximum), 1, 1)];
        [sedimentation_encoder endEncoding];

        if (!impl_->finish(command_buffer))
            return false;
        for (int index = 0; index < 6; ++index)
            std::swap(tendency_slots[index]->current, tendency_slots[index]->candidate);
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        for (int index = 0; index < 6; ++index)
        {
            std::memcpy(outputs_data[index], tendency_slots[index]->current.contents, field_bytes);
            impl_->resident.fields.note_device_write(tendency_names[index], outputs_data[index], extent);
            impl_->resident.fields.note_host_synced(tendency_names[index]);
        }
        ++impl_->telemetry.completed_kessler_pointwise_dispatches;
        ++impl_->telemetry.completed_kessler_sedimentation_dispatches;
        ++impl_->telemetry.completed_kessler_fused_dispatches;
        impl_->telemetry.kessler_resident_tendency_writes += 6;
        return true;
    }
}
