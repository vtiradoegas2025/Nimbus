TerrainComputeKernels* MetalComputeBackend::terrain_kernels()
{
    return this;
}

const TerrainComputeKernels* MetalComputeBackend::terrain_kernels() const
{
    return this;
}

namespace
{

template <std::size_t Count>
bool distinct_mapped_destinations(const std::array<float*, Count>& destinations, std::size_t bytes) noexcept
{
    if (bytes == 0)
        return false;
    for (std::size_t first = 0; first < Count; ++first)
    {
        if (destinations[first] == nullptr)
            return false;
        const auto first_address = reinterpret_cast<std::uintptr_t>(destinations[first]);
        for (std::size_t second = first + 1; second < Count; ++second)
        {
            if (destinations[second] == nullptr)
                return false;
            const auto second_address = reinterpret_cast<std::uintptr_t>(destinations[second]);
            const auto distance =
                first_address < second_address ? second_address - first_address : first_address - second_address;
            if (distance < bytes)
                return false;
        }
    }
    return true;
}

} // namespace

bool MetalComputeBackend::supports_mapped_mass_continuity() const
{
    return ready(KernelId::MappedTerrainMassContinuity);
}

bool MetalComputeBackend::dispatch_mapped_mass_continuity(
    const MappedTerrainMassContinuityDispatch& d)
{
    @autoreleasepool
    {
        const float* inputs[12] = {
            d.rho, d.u, d.v, d.w, d.p, d.jacobian,
            d.jacobian_x_positive_face, d.jacobian_y_positive_face,
            d.metric_x_positive_zeta_face, d.metric_y_positive_zeta_face,
            d.metric_x_ground_face, d.metric_y_ground_face};
        std::array<float*, 6> destinations = {
            d.ground_physical_w, d.drho_dt, d.dp_dt,
            d.mass_flux_x_positive, d.mass_flux_y_positive,
            d.mass_flux_zeta_positive};
        std::size_t cells = 0;
        std::size_t bytes = 0;
        if (!supports_mapped_mass_continuity() || d.report == nullptr ||
            d.nx < 3 || d.ny < 3 || d.nz < 1 ||
            !std::isfinite(d.dx) || d.dx <= 0.0f ||
            !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dzeta) || d.dzeta <= 0.0f ||
            !std::isfinite(d.gamma) || d.gamma <= 1.0f ||
            !std::isfinite(d.frame_u_ms) || !std::isfinite(d.frame_v_ms) ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) ||
            bytes > impl_->max_buffer_bytes() ||
            !distinct_mapped_destinations(destinations, bytes))
            return false;
        for (const float* input : inputs)
            if (input == nullptr)
                return false;
        for (std::size_t index = 0; index < cells; ++index)
        {
            const float state_values[] = {
                d.rho[index], d.u[index], d.v[index], d.w[index], d.p[index]};
            if (!std::all_of(std::begin(state_values), std::end(state_values),
                    [](float value) { return std::isfinite(value); }) ||
                d.rho[index] <= 0.0f || !std::isfinite(d.jacobian[index]) ||
                d.jacobian[index] <= 0.0f)
                return false;
        }

        id<MTLBuffer> input_buffers[12] = {};
        id<MTLBuffer> output_buffers[6] = {};
        for (int field = 0; field < 12; ++field)
        {
            input_buffers[field] = [impl_->hardware.device
                newBufferWithBytes:inputs[field]
                              length:bytes
                             options:MTLResourceStorageModeShared];
            if (input_buffers[field] == nil)
                return false;
        }
        for (int field = 0; field < 6; ++field)
        {
            output_buffers[field] = [impl_->hardware.device
                newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
            if (output_buffers[field] == nil)
                return false;
        }
        id<MTLBuffer> dummy = [impl_->hardware.device
            newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (dummy == nil)
            return false;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        const auto pipeline = impl_->pipeline(KernelId::MappedTerrainMassContinuity);
        const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
            return false;
        MappedTerrainMassContinuityConstants constants{};
        constants.nx = d.nx;
        constants.ny = d.ny;
        constants.nz = d.nz;
        constants.dx = d.dx;
        constants.dy = d.dy;
        constants.dzeta = d.dzeta;
        constants.gamma = d.gamma;
        constants.frame_u_ms = d.frame_u_ms;
        constants.frame_v_ms = d.frame_v_ms;
        const MTLSize grid = MTLSizeMake(cells, 1, 1);
        const MTLSize group = MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1);

        id<MTLComputeCommandEncoder> ground_encoder =
            [command_buffer computeCommandEncoder];
        if (ground_encoder == nil)
            return false;
        constants.stage = 0;
        [ground_encoder setComputePipelineState:pipeline];
        for (NSUInteger binding = 0; binding < 18; ++binding)
            [ground_encoder setBuffer:dummy offset:0 atIndex:binding];
        [ground_encoder setBuffer:input_buffers[1] offset:0 atIndex:0];
        [ground_encoder setBuffer:input_buffers[2] offset:0 atIndex:1];
        [ground_encoder setBuffer:input_buffers[3] offset:0 atIndex:2];
        [ground_encoder setBuffer:input_buffers[10] offset:0 atIndex:3];
        [ground_encoder setBuffer:input_buffers[11] offset:0 atIndex:4];
        [ground_encoder setBuffer:output_buffers[0] offset:0 atIndex:5];
        [ground_encoder setBytes:&constants length:sizeof(constants) atIndex:18];
        [ground_encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [ground_encoder endEncoding];

        id<MTLComputeCommandEncoder> continuity_encoder =
            [command_buffer computeCommandEncoder];
        if (continuity_encoder == nil)
            return false;
        constants.stage = 1;
        [continuity_encoder setComputePipelineState:pipeline];
        for (NSUInteger binding = 0; binding < 18; ++binding)
            [continuity_encoder setBuffer:dummy offset:0 atIndex:binding];
        for (NSUInteger binding = 0; binding < 10; ++binding)
            [continuity_encoder setBuffer:input_buffers[binding]
                                   offset:0 atIndex:binding];
        for (NSUInteger field = 1; field < 6; ++field)
            [continuity_encoder setBuffer:output_buffers[field]
                                   offset:0 atIndex:field + 9];
        [continuity_encoder setBytes:&constants length:sizeof(constants) atIndex:18];
        [continuity_encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [continuity_encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        impl_->telemetry.completed_mapped_mass_continuity_dispatches += 2;

        const float* candidate[6] = {};
        for (int field = 0; field < 6; ++field)
        {
            candidate[field] = static_cast<const float*>(output_buffers[field].contents);
            if (candidate[field] == nullptr)
                return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
            for (const float* field : candidate)
                if (!std::isfinite(field[index]))
                    return false;
        if (d.completed_observer != nullptr)
            d.completed_observer(d.completed_observer_context,
                candidate[0], candidate[1], candidate[2], candidate[3],
                candidate[4], candidate[5]);
        for (std::size_t field = 0; field < destinations.size(); ++field)
            std::memcpy(destinations[field], candidate[field], bytes);
        *d.report = {1u, 1u, true};
        return true;
    }
}

bool MetalComputeBackend::supports_mapped_fast_momentum() const
{
    return ready(KernelId::MappedTerrainFastMomentum);
}

bool MetalComputeBackend::dispatch_mapped_fast_momentum(
    const MappedTerrainFastMomentumDispatch& d)
{
    @autoreleasepool
    {
        const float* inputs[4] = {
            d.rho, d.p, d.physical_height, d.reference_pressure};
        std::array<float*, 3> destinations = {d.du_dt, d.dv_dt, d.dw_dt};
        std::size_t cells = 0;
        std::size_t bytes = 0;
        if (!supports_mapped_fast_momentum() || d.report == nullptr ||
            d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !std::isfinite(d.dx) || d.dx <= 0.0f ||
            !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.nominal_dz) || d.nominal_dz <= 0.0f ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) ||
            bytes > impl_->max_buffer_bytes() ||
            !distinct_mapped_destinations(destinations, bytes))
            return false;
        for (const float* input : inputs)
            if (input == nullptr)
                return false;
        for (std::size_t index = 0; index < cells; ++index)
            if (!std::isfinite(d.rho[index]) || d.rho[index] <= 1.0e-6f ||
                !std::isfinite(d.p[index]) ||
                !std::isfinite(d.physical_height[index]))
                return false;
        for (int level = 0; level < d.nz; ++level)
            if (!std::isfinite(d.reference_pressure[level]) ||
                d.reference_pressure[level] <= 0.0f)
                return false;

        if (d.sampled_reference_state)
        {
            if (cells > (std::numeric_limits<std::uint32_t>::max() - std::size_t(d.nz)) / 3)
                return false;
            for (std::size_t n = 0; n < cells; ++n)
                for (std::size_t field = 0; field < 3; ++field)
                {
                    const float value = d.reference_pressure[d.nz + 3 * n + field];
                    if (!std::isfinite(value) || (field < 2 ? value <= 0 : value < 0))
                        return false;
                }
        }
        id<MTLBuffer> input_buffers[4] = {};
        id<MTLBuffer> output_buffers[3] = {};
        for (int field = 0; field < 4; ++field)
        {
            const std::size_t input_bytes = field == 3
                ? (static_cast<std::size_t>(d.nz) + (d.sampled_reference_state ? 3 * cells : 0)) * sizeof(float) : bytes;
            if (input_bytes > impl_->max_buffer_bytes())
                return false;
            input_buffers[field] = [impl_->hardware.device
                newBufferWithBytes:inputs[field]
                              length:input_bytes
                             options:MTLResourceStorageModeShared];
            if (input_buffers[field] == nil)
                return false;
        }
        for (int field = 0; field < 3; ++field)
        {
            output_buffers[field] = [impl_->hardware.device
                newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
            if (output_buffers[field] == nil)
                return false;
        }
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        const auto pipeline = impl_->pipeline(KernelId::MappedTerrainFastMomentum);
        const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
        if (encoder == nil || maximum == 0)
            return false;
        MappedTerrainFastMomentumConstants constants{};
        constants.sampled_reference = d.sampled_reference_state ? 1 : 0;
        constants.nx = d.nx;
        constants.ny = d.ny;
        constants.nz = d.nz;
        constants.dx = d.dx;
        constants.dy = d.dy;
        constants.nominal_dz = d.nominal_dz;
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger field = 0; field < 4; ++field)
            [encoder setBuffer:input_buffers[field] offset:0 atIndex:field];
        for (NSUInteger field = 0; field < 3; ++field)
            [encoder setBuffer:output_buffers[field] offset:0 atIndex:field + 4];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:7];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        ++impl_->telemetry.completed_mapped_fast_momentum_dispatches;

        const float* candidate[3] = {};
        for (int field = 0; field < 3; ++field)
        {
            candidate[field] = static_cast<const float*>(output_buffers[field].contents);
            if (candidate[field] == nullptr)
                return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
            for (const float* field : candidate)
                if (!std::isfinite(field[index]))
                    return false;
        if (d.completed_observer != nullptr)
            d.completed_observer(d.completed_observer_context,
                candidate[0], candidate[1], candidate[2]);
        for (std::size_t field = 0; field < destinations.size(); ++field)
            std::memcpy(destinations[field], candidate[field], bytes);
        *d.report = {1u, true};
        return true;
    }
}

bool MetalComputeBackend::supports_mapped_slow_tendencies() const
{
    return ready(KernelId::MappedTerrainSlowTendencies);
}

bool MetalComputeBackend::dispatch_mapped_slow_tendencies(
    const MappedTerrainSlowTendenciesDispatch& d)
{
    @autoreleasepool
    {
        const bool required_pointers = d.u && d.v && d.w && d.rho && d.p &&
            d.physical_height && d.jacobian_x_positive_face &&
            d.jacobian_y_positive_face && d.metric_x_positive_zeta_face &&
            d.metric_y_positive_zeta_face && d.ground_physical_w &&
            d.reference_pressure && d.reference_theta &&
            (!d.has_moisture || (d.qv && d.reference_qv)) &&
            (!d.has_loading || d.hydrometeor_loading) && d.du_dt && d.dv_dt &&
            d.dw_dt && d.drho_dt && d.dp_dt && d.du_advection_dt &&
            d.dv_advection_dt && d.dw_advection_dt && d.report;
        std::array<float*, 8> destinations = {
            d.du_dt, d.dv_dt, d.dw_dt, d.drho_dt, d.dp_dt,
            d.du_advection_dt, d.dv_advection_dt, d.dw_advection_dt};
        std::size_t cells = 0;
        std::size_t bytes = 0;
        if (!supports_mapped_slow_tendencies() || !required_pointers ||
            d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !std::isfinite(d.dx) || d.dx <= 0.0f ||
            !std::isfinite(d.dy) || d.dy <= 0.0f ||
            !std::isfinite(d.dzeta) || d.dzeta <= 0.0f ||
            !std::isfinite(d.nominal_dz) || d.nominal_dz <= 0.0f ||
            !std::isfinite(d.gravity) || d.gravity <= 0.0f ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) ||
            bytes > impl_->max_buffer_bytes() ||
            !distinct_mapped_destinations(destinations, bytes))
            return false;
        for (std::size_t index = 0; index < cells; ++index)
        {
            const float values[] = {
                d.u[index], d.v[index], d.w[index], d.rho[index], d.p[index],
                d.physical_height[index], d.jacobian_x_positive_face[index],
                d.jacobian_y_positive_face[index],
                d.metric_x_positive_zeta_face[index],
                d.metric_y_positive_zeta_face[index], d.ground_physical_w[index]};
            const int j = static_cast<int>((index / static_cast<std::size_t>(d.nz)) %
                                           static_cast<std::size_t>(d.ny));
            const int i = static_cast<int>(index /
                (static_cast<std::size_t>(d.ny) * static_cast<std::size_t>(d.nz)));
            const bool active_column = i > 0 && i < d.nx - 1 &&
                j > 0 && j < d.ny - 1;
            if (!std::all_of(std::begin(values), std::end(values),
                    [](float value) { return std::isfinite(value); }) ||
                d.rho[index] <= 1.0e-6f ||
                (active_column && (d.jacobian_x_positive_face[index] <= 0.0f ||
                                   d.jacobian_y_positive_face[index] <= 0.0f)) ||
                (d.has_moisture && (!std::isfinite(d.qv[index]) || d.qv[index] < 0.0f)) ||
                (d.has_loading && (!std::isfinite(d.hydrometeor_loading[index]) ||
                                   d.hydrometeor_loading[index] < 0.0f)))
                return false;
        }
        for (int level = 0; level < d.nz; ++level)
            if (!std::isfinite(d.reference_pressure[level]) ||
                d.reference_pressure[level] <= 0.0f ||
                !std::isfinite(d.reference_theta[level]) ||
                d.reference_theta[level] <= 0.0f ||
                (d.has_moisture && (!std::isfinite(d.reference_qv[level]) ||
                                    d.reference_qv[level] < 0.0f)))
                return false;

        std::vector<float> disabled_field;
        std::vector<float> disabled_profile;
        if (!d.has_moisture || !d.has_loading)
            disabled_field.assign(cells, 0.0f);
        if (!d.has_moisture)
            disabled_profile.assign(static_cast<std::size_t>(d.nz), 0.0f);
        const float* inputs[16] = {
            d.u, d.v, d.w, d.rho, d.p, d.physical_height,
            d.jacobian_x_positive_face, d.jacobian_y_positive_face,
            d.metric_x_positive_zeta_face, d.metric_y_positive_zeta_face,
            d.ground_physical_w, d.reference_pressure, d.reference_theta,
            d.has_moisture ? d.qv : disabled_field.data(),
            d.has_moisture ? d.reference_qv : disabled_profile.data(),
            d.has_loading ? d.hydrometeor_loading : disabled_field.data()};
        if (d.sampled_reference_state)
        {
            if (cells > (std::numeric_limits<std::uint32_t>::max() - std::size_t(d.nz)) / 3)
                return false;
            for (std::size_t n = 0; n < cells; ++n)
                for (std::size_t field = 0; field < 3; ++field)
                {
                    const float value = d.reference_pressure[d.nz + 3 * n + field];
                    if (!std::isfinite(value) || (field < 2 ? value <= 0 : value < 0))
                        return false;
                }
        }
        id<MTLBuffer> input_buffers[16] = {};
        id<MTLBuffer> output_buffers[8] = {};
        for (int field = 0; field < 16; ++field)
        {
            const bool profile = field == 11 || field == 12 || field == 14;
            const std::size_t input_bytes = profile
                ? (static_cast<std::size_t>(d.nz) + (field == 11 && d.sampled_reference_state ? 3 * cells : 0)) * sizeof(float) : bytes;
            if (input_bytes > impl_->max_buffer_bytes())
                return false;
            input_buffers[field] = [impl_->hardware.device
                newBufferWithBytes:inputs[field]
                              length:input_bytes
                             options:MTLResourceStorageModeShared];
            if (input_buffers[field] == nil)
                return false;
        }
        for (int field = 0; field < 8; ++field)
        {
            output_buffers[field] = [impl_->hardware.device
                newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
            if (output_buffers[field] == nil)
                return false;
        }
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        const auto pipeline = impl_->pipeline(KernelId::MappedTerrainSlowTendencies);
        const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
        if (encoder == nil || maximum == 0)
            return false;
        MappedTerrainSlowTendenciesConstants constants{};
        constants.sampled_reference = d.sampled_reference_state ? 1 : 0;
        constants.nx = d.nx;
        constants.ny = d.ny;
        constants.nz = d.nz;
        constants.has_moisture = d.has_moisture ? 1 : 0;
        constants.has_loading = d.has_loading ? 1 : 0;
        constants.dx = d.dx;
        constants.dy = d.dy;
        constants.dzeta = d.dzeta;
        constants.nominal_dz = d.nominal_dz;
        constants.gravity = d.gravity;
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger field = 0; field < 16; ++field)
            [encoder setBuffer:input_buffers[field] offset:0 atIndex:field];
        for (NSUInteger field = 0; field < 8; ++field)
            [encoder setBuffer:output_buffers[field] offset:0 atIndex:field + 16];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:24];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        ++impl_->telemetry.completed_mapped_slow_tendencies_dispatches;

        const float* candidate[8] = {};
        for (int field = 0; field < 8; ++field)
        {
            candidate[field] = static_cast<const float*>(output_buffers[field].contents);
            if (candidate[field] == nullptr)
                return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
            for (const float* field : candidate)
                if (!std::isfinite(field[index]))
                    return false;
        if (d.completed_observer != nullptr)
            d.completed_observer(d.completed_observer_context,
                candidate[0], candidate[1], candidate[2], candidate[3],
                candidate[4], candidate[5], candidate[6], candidate[7]);
        for (std::size_t field = 0; field < destinations.size(); ++field)
            std::memcpy(destinations[field], candidate[field], bytes);
        *d.report = {1u, true, true};
        return true;
    }
}

bool MetalComputeBackend::supports_mapped_scalar_transport() const
{
    return ready(KernelId::AdvectConservativeScalarFluxCartesianCgrid) &&
        ready(KernelId::MappedTerrainScalarLimiter) &&
        ready(KernelId::MappedTerrainScalarUpdate);
}

bool MetalComputeBackend::dispatch_mapped_scalar_transport(
    const MappedTerrainScalarTransportDispatch& d)
{
    @autoreleasepool
    {
        const float* inputs[7] = {
            d.scalar_begin, d.density_begin, d.density_end,
            d.mass_flux_x, d.mass_flux_y, d.mass_flux_zeta, d.jacobian};
        std::size_t cells = 0;
        std::size_t bytes = 0;
        if ((d.preserve_positive_support && !d.positive_definite) || !supports_mapped_scalar_transport() ||
            d.scalar_end == nullptr || d.report == nullptr || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !std::isfinite(d.dx) || d.dx <= 0.0f || !std::isfinite(d.dy) || d.dy <= 0.0f || !std::isfinite(d.dzeta) ||
            d.dzeta <= 0.0f || !std::isfinite(d.dt) || d.dt <= 0.0f || !std::isfinite(d.density_floor) ||
            d.density_floor <= 0.0f || !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) ||
            bytes > impl_->max_buffer_bytes())
            return false;
        for (const float* input : inputs)
            if (input == nullptr)
                return false;
        for (std::size_t index = 0; index < cells; ++index)
        {
            for (const float* input : inputs)
                if (!std::isfinite(input[index]))
                    return false;
            if (d.density_begin[index] <= 0.0f || d.density_end[index] <= 0.0f ||
                d.jacobian[index] <= 0.0f)
                return false;
        }

        id<MTLBuffer> input_buffers[7] = {};
        for (int field = 0; field < 7; ++field)
        {
            input_buffers[field] = [impl_->hardware.device
                newBufferWithBytes:inputs[field]
                              length:bytes
                             options:MTLResourceStorageModeShared];
            if (input_buffers[field] == nil)
                return false;
        }
        id<MTLBuffer> scalar_flux[3] = {};
        for (int field = 0; field < 3; ++field)
        {
            scalar_flux[field] = [impl_->hardware.device
                newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
            if (scalar_flux[field] == nil)
                return false;
        }
        std::vector<float> limiter_seed(cells, 1.0f);
        id<MTLBuffer> limiter = [impl_->hardware.device
            newBufferWithBytes:limiter_seed.data()
                          length:bytes
                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> candidate = [impl_->hardware.device
            newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> applied[5] = {};
        for (auto& field : applied)
        {
            field = [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (field == nil)
                return false;
        }
        if (limiter == nil || candidate == nil)
            return false;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        const MTLSize grid = MTLSizeMake(cells, 1, 1);

        id<MTLComputeCommandEncoder> flux_encoder =
            [command_buffer computeCommandEncoder];
        const auto flux_pipeline = impl_->pipeline(
            KernelId::AdvectConservativeScalarFluxCartesianCgrid);
        const NSUInteger flux_maximum = flux_pipeline.maxTotalThreadsPerThreadgroup;
        if (flux_encoder == nil || flux_maximum == 0)
            return false;
        MappedTerrainScalarFluxConstants flux_constants{};
        flux_constants.nx = d.nx;
        flux_constants.ny = d.ny;
        flux_constants.nz   = d.nz;
        flux_constants.pad0 = (d.weno ? 0 : 1) | (d.preserve_positive_support ? 2 : 0);
        [flux_encoder setComputePipelineState:flux_pipeline];
        [flux_encoder setBuffer:input_buffers[0] offset:0 atIndex:0];
        [flux_encoder setBuffer:input_buffers[3] offset:0 atIndex:1];
        [flux_encoder setBuffer:input_buffers[4] offset:0 atIndex:2];
        [flux_encoder setBuffer:input_buffers[5] offset:0 atIndex:3];
        for (NSUInteger field = 0; field < 3; ++field)
            [flux_encoder setBuffer:scalar_flux[field] offset:0 atIndex:field + 4];
        [flux_encoder setBytes:&flux_constants length:sizeof(flux_constants) atIndex:7];
        [flux_encoder dispatchThreads:grid
            threadsPerThreadgroup:MTLSizeMake(
                std::min<NSUInteger>(64, flux_maximum), 1, 1)];
        [flux_encoder endEncoding];

        if (d.positive_definite)
        {
            id<MTLComputeCommandEncoder> limiter_encoder =
                [command_buffer computeCommandEncoder];
            const auto limiter_pipeline = impl_->pipeline(
                KernelId::MappedTerrainScalarLimiter);
            const NSUInteger limiter_maximum =
                limiter_pipeline.maxTotalThreadsPerThreadgroup;
            if (limiter_encoder == nil || limiter_maximum == 0)
                return false;
            MappedTerrainScalarLimiterConstants limiter_constants{};
            limiter_constants.nx = d.nx;
            limiter_constants.ny = d.ny;
            limiter_constants.nz        = d.nz;
            limiter_constants.padding0  = d.preserve_positive_support ? 1 : 0;
            limiter_constants.inv_dx    = 1.0f / d.dx;
            limiter_constants.inv_dy = 1.0f / d.dy;
            limiter_constants.inv_dzeta = 1.0f / d.dzeta;
            limiter_constants.dt = d.dt;
            [limiter_encoder setComputePipelineState:limiter_pipeline];
            [limiter_encoder setBuffer:input_buffers[0] offset:0 atIndex:0];
            [limiter_encoder setBuffer:input_buffers[1] offset:0 atIndex:1];
            for (NSUInteger field = 0; field < 3; ++field)
                [limiter_encoder setBuffer:scalar_flux[field]
                                     offset:0 atIndex:field + 2];
            [limiter_encoder setBuffer:input_buffers[6] offset:0 atIndex:5];
            [limiter_encoder setBuffer:limiter offset:0 atIndex:6];
            [limiter_encoder setBytes:&limiter_constants
                               length:sizeof(limiter_constants) atIndex:7];
            [limiter_encoder dispatchThreads:grid
                threadsPerThreadgroup:MTLSizeMake(
                    std::min<NSUInteger>(64, limiter_maximum), 1, 1)];
            [limiter_encoder endEncoding];
        }

        id<MTLComputeCommandEncoder> update_encoder =
            [command_buffer computeCommandEncoder];
        const auto update_pipeline = impl_->pipeline(KernelId::MappedTerrainScalarUpdate);
        const NSUInteger update_maximum = update_pipeline.maxTotalThreadsPerThreadgroup;
        if (update_encoder == nil || update_maximum == 0)
            return false;
        MappedTerrainScalarUpdateConstants update_constants{};
        update_constants.nx = d.nx;
        update_constants.ny = d.ny;
        update_constants.nz = d.nz;
        update_constants.positive_definite = d.positive_definite ? 1 : 0;
        update_constants.inv_dx = 1.0f / d.dx;
        update_constants.inv_dy = 1.0f / d.dy;
        update_constants.inv_dzeta = 1.0f / d.dzeta;
        update_constants.dt = d.dt;
        update_constants.density_floor = d.density_floor;
        [update_encoder setComputePipelineState:update_pipeline];
        [update_encoder setBuffer:input_buffers[0] offset:0 atIndex:0];
        [update_encoder setBuffer:input_buffers[1] offset:0 atIndex:1];
        [update_encoder setBuffer:input_buffers[2] offset:0 atIndex:2];
        for (NSUInteger field = 0; field < 3; ++field)
            [update_encoder setBuffer:scalar_flux[field] offset:0 atIndex:field + 3];
        [update_encoder setBuffer:input_buffers[6] offset:0 atIndex:6];
        [update_encoder setBuffer:limiter offset:0 atIndex:7];
        [update_encoder setBuffer:candidate offset:0 atIndex:8];
        for (NSUInteger field = 0; field < 5; ++field)
            [update_encoder setBuffer:applied[field] offset:0 atIndex:9 + field];
        [update_encoder setBytes:&update_constants
                          length:sizeof(update_constants) atIndex:14];
        [update_encoder dispatchThreads:grid
            threadsPerThreadgroup:MTLSizeMake(
                std::min<NSUInteger>(64, update_maximum), 1, 1)];
        [update_encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        ++impl_->telemetry.completed_conservative_cgrid_scalars;
        ++impl_->telemetry.completed_conservative_cgrid_scalar_flux_dispatches;
        ++impl_->telemetry.completed_mapped_scalar_update_dispatches;
        if (d.positive_definite)
            ++impl_->telemetry.completed_mapped_scalar_limiter_dispatches;

        const float* flux_result[3] = {
            static_cast<const float*>(scalar_flux[0].contents),
            static_cast<const float*>(scalar_flux[1].contents),
            static_cast<const float*>(scalar_flux[2].contents)};
        const float* limiter_result = static_cast<const float*>(limiter.contents);
        const float* candidate_result = static_cast<const float*>(candidate.contents);
        if (limiter_result == nullptr || candidate_result == nullptr ||
            flux_result[0] == nullptr || flux_result[1] == nullptr ||
            flux_result[2] == nullptr)
            return false;
        for (std::size_t index = 0; index < cells; ++index)
        {
            if (!std::isfinite(candidate_result[index]) ||
                !std::isfinite(limiter_result[index]) || limiter_result[index] < 0.0f ||
                limiter_result[index] > 1.0f)
                return false;
            for (const float* field : flux_result)
                if (!std::isfinite(field[index]))
                    return false;
        }
        const float* applied_result[5] = {};
        for (int field = 0; field < 5; ++field)
        {
            applied_result[field] = static_cast<const float*>(applied[field].contents);
            if (!applied_result[field] || !std::all_of(applied_result[field], applied_result[field] + cells,
                    [](float value) { return std::isfinite(value); }))
                return false;
        }
        if (d.applied_observer)
            d.applied_observer(d.applied_observer_context, candidate_result, applied_result[0],
                applied_result[1], applied_result[2], applied_result[3], applied_result[4]);
        if (d.completed_observer != nullptr)
            d.completed_observer(d.completed_observer_context,
                candidate_result, flux_result[0], flux_result[1], flux_result[2],
                limiter_result);
        std::memcpy(d.scalar_end, candidate_result, bytes);
        *d.report = {1u, d.positive_definite ? 1u : 0u, 1u, true};
        return true;
    }
}

bool MetalComputeBackend::supports_mapped_final_state() const
{
    return ready(KernelId::MappedTerrainEos) &&
        ready(KernelId::MappedTerrainBoundary);
}

bool MetalComputeBackend::dispatch_mapped_final_state(
    const MappedTerrainFinalStateDispatch& d)
{
    @autoreleasepool
    {
        const float* inputs[6] = {d.u, d.v, d.w, d.rho, d.p, d.theta};
        std::array<float*, 5> destinations = {
            d.u_final, d.v_final, d.w_final, d.rho_final, d.p_final};
        std::size_t cells = 0;
        std::size_t bytes = 0;
        if (!supports_mapped_final_state() || !nimbus::compute::mapped_boundary::valid_options(d) || d.report == nullptr ||
            d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) ||
            bytes > impl_->max_buffer_bytes() ||
            !distinct_mapped_destinations(destinations, bytes))
            return false;
        for (const float* input : inputs)
            if (input == nullptr)
                return false;
        for (std::size_t index = 0; index < cells; ++index)
        {
            for (const float* input : inputs)
                if (!std::isfinite(input[index]))
                    return false;
            if (d.rho[index] <= 0.0f || d.p[index] <= 0.0f ||
                d.theta[index] <= 0.0f)
                return false;
        }

        id<MTLBuffer> input_buffers[5] = {};
        for (int field = 0; field < 5; ++field)
        {
            input_buffers[field] = [impl_->hardware.device
                newBufferWithBytes:inputs[field]
                              length:bytes
                             options:MTLResourceStorageModeShared];
            if (input_buffers[field] == nil)
                return false;
        }
        id<MTLBuffer> theta_buffer = [impl_->hardware.device
            newBufferWithBytes:d.theta length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> eos_pressure = [impl_->hardware.device
            newBufferWithBytes:d.p length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffers[5] = {};
        if (theta_buffer == nil || eos_pressure == nil)
            return false;
        for (int field = 0; field < 5; ++field)
        {
            output_buffers[field] = [impl_->hardware.device
                newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
            if (output_buffers[field] == nil)
                return false;
        }
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        const MTLSize grid = MTLSizeMake(cells, 1, 1);
        MappedTerrainEosConstants eos_constants{};
        eos_constants.nx = d.nx; eos_constants.ny = d.ny; eos_constants.nz = d.nz;
        if (d.apply_eos)
        {
            id<MTLComputeCommandEncoder> eos_encoder =
                [command_buffer computeCommandEncoder];
            const auto eos_pipeline = impl_->pipeline(KernelId::MappedTerrainEos);
            const NSUInteger eos_maximum = eos_pipeline.maxTotalThreadsPerThreadgroup;
            if (eos_encoder == nil || eos_maximum == 0)
                return false;
            [eos_encoder setComputePipelineState:eos_pipeline];
            [eos_encoder setBuffer:input_buffers[3] offset:0 atIndex:0];
            [eos_encoder setBuffer:theta_buffer offset:0 atIndex:1];
            [eos_encoder setBuffer:eos_pressure offset:0 atIndex:2];
            [eos_encoder setBytes:&eos_constants length:sizeof(eos_constants) atIndex:3];
            [eos_encoder dispatchThreads:grid
                threadsPerThreadgroup:MTLSizeMake(
                    std::min<NSUInteger>(64, eos_maximum), 1, 1)];
            [eos_encoder endEncoding];
        }

        id<MTLComputeCommandEncoder> boundary_encoder =
            [command_buffer computeCommandEncoder];
        const auto boundary_pipeline = impl_->pipeline(KernelId::MappedTerrainBoundary);
        const NSUInteger boundary_maximum =
            boundary_pipeline.maxTotalThreadsPerThreadgroup;
        if (boundary_encoder == nil || boundary_maximum == 0)
            return false;
        const auto boundary_constants = nimbus::compute::mapped_boundary::constants(d);
        [boundary_encoder setComputePipelineState:boundary_pipeline];
        for (NSUInteger field = 0; field < 4; ++field)
            [boundary_encoder setBuffer:input_buffers[field] offset:0 atIndex:field];
        [boundary_encoder setBuffer:eos_pressure offset:0 atIndex:4];
        for (NSUInteger field = 0; field < 5; ++field)
            [boundary_encoder setBuffer:output_buffers[field] offset:0 atIndex:field + 5];
        [boundary_encoder setBytes:&boundary_constants
                            length:sizeof(boundary_constants) atIndex:10];
        [boundary_encoder dispatchThreads:grid
            threadsPerThreadgroup:MTLSizeMake(
                std::min<NSUInteger>(64, boundary_maximum), 1, 1)];
        [boundary_encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        if (d.apply_eos)
            ++impl_->telemetry.completed_mapped_eos_dispatches;
        ++impl_->telemetry.completed_mapped_boundary_dispatches;

        const float* candidate[5] = {};
        for (int field = 0; field < 5; ++field)
        {
            candidate[field] = static_cast<const float*>(output_buffers[field].contents);
            if (candidate[field] == nullptr)
                return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
            for (const float* field : candidate)
                if (!std::isfinite(field[index]))
                    return false;
        if (d.completed_observer != nullptr)
            d.completed_observer(d.completed_observer_context,
                candidate[0], candidate[1], candidate[2], candidate[3], candidate[4]);
        for (std::size_t field = 0; field < destinations.size(); ++field)
            std::memcpy(destinations[field], candidate[field], bytes);
        *d.report = {d.apply_eos ? 1u : 0u, 1u, true};
        return true;
    }
}

bool MetalComputeBackend::supports_mapped_state_update() const
{
    return ready(KernelId::MappedTerrainStateUpdate);
}

bool MetalComputeBackend::dispatch_mapped_state_update(const MappedTerrainStateUpdateDispatch& d)
{
    @autoreleasepool
    {
        const float* inputs[11] = {d.u, d.v, d.w, d.rho, d.p, d.theta, d.du_dt, d.dv_dt, d.dw_dt, d.drho_dt, d.dp_dt};
        std::array<float*, 5> destinations = {
            d.u_candidate, d.v_candidate, d.w_candidate, d.rho_candidate, d.p_candidate};
        const auto  mode  = static_cast<std::uint32_t>(d.mode);
        std::size_t cells = 0;
        std::size_t bytes = 0;
        if (!supports_mapped_state_update() || d.report == nullptr || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            mode > static_cast<std::uint32_t>(MappedTerrainStateUpdateMode::FastMomentum) || !std::isfinite(d.dt) ||
            d.dt <= 0.0f || !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || bytes > impl_->max_buffer_bytes() ||
            !distinct_mapped_destinations(destinations, bytes))
            return false;
        for (const float* input : inputs)
            if (input == nullptr)
                return false;
        for (std::size_t index = 0; index < cells; ++index)
        {
            for (const float* input : inputs)
                if (!std::isfinite(input[index]))
                    return false;
            if (d.rho[index] <= 0.0f || d.p[index] <= 0.0f || d.theta[index] <= 0.0f)
                return false;
        }

        id<MTLBuffer> input_buffers[11] = {};
        id<MTLBuffer> output_buffers[5] = {};
        for (int field = 0; field < 11; ++field)
        {
            input_buffers[field] = [impl_->hardware.device newBufferWithBytes:inputs[field]
                                                                       length:bytes
                                                                      options:MTLResourceStorageModeShared];
            if (input_buffers[field] == nil)
                return false;
        }
        for (int field = 0; field < 5; ++field)
        {
            output_buffers[field] =
                [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (output_buffers[field] == nil)
                return false;
        }

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        const NSUInteger maximum = impl_->pipeline(KernelId::MappedTerrainStateUpdate).maxTotalThreadsPerThreadgroup;
        if (encoder == nil || maximum == 0)
            return false;
        MappedTerrainStateUpdateConstants constants{};
        constants.nx   = d.nx;
        constants.ny   = d.ny;
        constants.nz   = d.nz;
        constants.mode = mode;
        constants.dt   = d.dt;
        [encoder setComputePipelineState:impl_->pipeline(KernelId::MappedTerrainStateUpdate)];
        for (NSUInteger field = 0; field < 11; ++field)
            [encoder setBuffer:input_buffers[field] offset:0 atIndex:field];
        for (NSUInteger field = 0; field < 5; ++field)
            [encoder setBuffer:output_buffers[field] offset:0 atIndex:field + 11];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:16];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        ++impl_->telemetry.completed_mapped_state_update_dispatches;

        const float* candidate[5] = {};
        for (int field = 0; field < 5; ++field)
        {
            candidate[field] = static_cast<const float*>(output_buffers[field].contents);
            if (candidate[field] == nullptr)
                return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
        {
            for (const float* field : candidate)
                if (!std::isfinite(field[index]))
                    return false;
            if (candidate[3][index] <= 0.0f || candidate[4][index] <= 0.0f)
                return false;
        }
        if (d.completed_observer != nullptr)
            d.completed_observer(
                d.completed_observer_context, candidate[0], candidate[1], candidate[2], candidate[3], candidate[4]);
        for (std::size_t field = 0; field < destinations.size(); ++field)
            std::memcpy(destinations[field], candidate[field], bytes);
        *d.report = {1u, d.mode, true};
        return true;
    }
}

bool MetalComputeBackend::supports_mapped_flux_accumulation() const
{
    return ready(KernelId::MappedTerrainFluxAccumulation);
}

bool MetalComputeBackend::dispatch_mapped_flux_accumulation(const MappedTerrainFluxAccumulationDispatch& d)
{
    @autoreleasepool
    {
        const float* inputs[6] = {
            d.accumulated_x, d.accumulated_y, d.accumulated_zeta, d.step_x, d.step_y, d.step_zeta};
        std::array<float*, 3> destinations = {
            d.accumulated_x_candidate, d.accumulated_y_candidate, d.accumulated_zeta_candidate};
        std::size_t cells = 0;
        std::size_t bytes = 0;
        if (!supports_mapped_flux_accumulation() || d.report == nullptr || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
            !std::isfinite(d.time_weight) || d.time_weight <= 0.0f || d.time_weight > 1.0f ||
            !checked_bytes(d.nx, d.ny, d.nz, cells, bytes) || bytes > impl_->max_buffer_bytes() ||
            !distinct_mapped_destinations(destinations, bytes))
            return false;
        for (const float* input : inputs)
            if (input == nullptr)
                return false;
        for (std::size_t index = 0; index < cells; ++index)
            for (const float* input : inputs)
                if (!std::isfinite(input[index]))
                    return false;

        id<MTLBuffer> input_buffers[6]  = {};
        id<MTLBuffer> output_buffers[3] = {};
        for (int field = 0; field < 6; ++field)
        {
            input_buffers[field] = [impl_->hardware.device newBufferWithBytes:inputs[field]
                                                                       length:bytes
                                                                      options:MTLResourceStorageModeShared];
            if (input_buffers[field] == nil)
                return false;
        }
        for (int field = 0; field < 3; ++field)
        {
            output_buffers[field] =
                [impl_->hardware.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (output_buffers[field] == nil)
                return false;
        }

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        const NSUInteger             maximum =
            impl_->pipeline(KernelId::MappedTerrainFluxAccumulation).maxTotalThreadsPerThreadgroup;
        if (encoder == nil || maximum == 0)
            return false;
        MappedTerrainFluxAccumulationConstants constants{};
        constants.nx          = d.nx;
        constants.ny          = d.ny;
        constants.nz          = d.nz;
        constants.time_weight = d.time_weight;
        [encoder setComputePipelineState:impl_->pipeline(KernelId::MappedTerrainFluxAccumulation)];
        for (NSUInteger field = 0; field < 6; ++field)
            [encoder setBuffer:input_buffers[field] offset:0 atIndex:field];
        for (NSUInteger field = 0; field < 3; ++field)
            [encoder setBuffer:output_buffers[field] offset:0 atIndex:field + 6];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:9];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        ++impl_->telemetry.completed_mapped_flux_accumulation_dispatches;

        const float* candidate[3] = {};
        for (int field = 0; field < 3; ++field)
        {
            candidate[field] = static_cast<const float*>(output_buffers[field].contents);
            if (candidate[field] == nullptr)
                return false;
        }
        for (std::size_t index = 0; index < cells; ++index)
            for (const float* field : candidate)
                if (!std::isfinite(field[index]))
                    return false;
        if (d.completed_observer != nullptr)
            d.completed_observer(d.completed_observer_context, candidate[0], candidate[1], candidate[2]);
        for (std::size_t field = 0; field < destinations.size(); ++field)
            std::memcpy(destinations[field], candidate[field], bytes);
        *d.report = {1u, true};
        return true;
    }
}
