// Cartesian scalar, vertical-flux, and conservative C-grid advection.
AdvectionComputeKernels* MetalComputeBackend::advection_kernels()
{
    return supports_advection_x() && supports_advection_y() ? this : nullptr;
}

const AdvectionComputeKernels* MetalComputeBackend::advection_kernels() const
{
    return supports_advection_x() && supports_advection_y() ? this : nullptr;
}

bool MetalComputeBackend::supports(AdvectionKernelOp op) const
{
    if (op == AdvectionKernelOp::AdvectionX)
        return supports_advection_x();
    if (op == AdvectionKernelOp::AdvectionY)
        return supports_advection_y();
    if (op == AdvectionKernelOp::BatchedAdvection)
        return supports_advection_pre_vertical_batch();
    if (op == AdvectionKernelOp::VerticalFlux)
        return supports_vertical_flux_dispatch();
    if (op == AdvectionKernelOp::ConservativeCartesianCgridTransport)
        return supports_conservative_cgrid_transport();
    return false;
}

bool MetalComputeBackend::supports_advection_x() const noexcept
{
    return ready(nimbus::compute::KernelId::AdvectX);
}

bool MetalComputeBackend::supports_advection_y() const noexcept
{
    return ready(nimbus::compute::KernelId::AdvectY);
}

bool MetalComputeBackend::supports_advection_pre_vertical_batch() const noexcept
{
    return supports_advection_x() && supports_advection_y();
}

bool MetalComputeBackend::supports_vertical_flux_dispatch() const noexcept
{
    return ready(nimbus::compute::KernelId::TvdVerticalFlux);
}

bool MetalComputeBackend::supports_conservative_cgrid_transport() const noexcept
{
    return ready(nimbus::compute::KernelId::AdvectConservativeMassFluxCartesianCgrid) &&
           ready(nimbus::compute::KernelId::AdvectConservativeUpdateCartesianCgrid);
}

bool MetalComputeBackend::supports_supplied_cartesian_cgrid_mass_flux() const
{
    return supports_conservative_cgrid_transport();
}

bool MetalComputeBackend::dispatch_advection_x(const ScalarAdvectionDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        std::size_t interior_cells = 0;
        std::size_t interior_bytes = 0;
        if (!supports_advection_x() || dispatch.src == nullptr || dispatch.velocity == nullptr ||
            dispatch.dst == nullptr || dispatch.dst == dispatch.velocity || dispatch.nr < 3 ||
            dispatch.nth < 1 || dispatch.nz < 3 || !std::isfinite(dispatch.primary_spacing) ||
            dispatch.primary_spacing <= 0.0f || !std::isfinite(dispatch.dt) || dispatch.dt <= 0.0f ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            !checked_bytes(dispatch.nr - 2, dispatch.nth, dispatch.nz - 2,
                           interior_cells, interior_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        id<MTLBuffer> source = [impl_->hardware.device newBufferWithBytes:dispatch.src
                                                          length:field_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> velocity = [impl_->hardware.device newBufferWithBytes:dispatch.velocity
                                                            length:field_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> destination = [impl_->hardware.device newBufferWithBytes:dispatch.src
                                                               length:field_bytes
                                                              options:MTLResourceStorageModeShared];
        if (source == nil || velocity == nil || destination == nil)
            return false;
        AdvectionXConstants constants{};
        constants.nr = dispatch.nr;
        constants.nth = dispatch.nth;
        constants.nz = dispatch.nz;
        constants.dx = dispatch.primary_spacing;
        constants.dt = dispatch.dt;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::AdvectX).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectX)];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:velocity offset:0 atIndex:1];
        [encoder setBuffer:destination offset:0 atIndex:2];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(interior_cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        std::memcpy(dispatch.dst, destination.contents, field_bytes);
        ++impl_->telemetry.completed_advection_x_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_advection_y(const ScalarAdvectionDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        std::size_t interior_cells = 0;
        std::size_t interior_bytes = 0;
        if (!supports_advection_y() || dispatch.src == nullptr || dispatch.velocity == nullptr ||
            dispatch.dst == nullptr || dispatch.dst == dispatch.velocity || dispatch.nr < 1 ||
            dispatch.nth < 3 || dispatch.nz < 3 || !std::isfinite(dispatch.primary_spacing) ||
            dispatch.primary_spacing <= 0.0f || !std::isfinite(dispatch.dt) || dispatch.dt <= 0.0f ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            !checked_bytes(dispatch.nr, dispatch.nth - 2, dispatch.nz - 2,
                           interior_cells, interior_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        id<MTLBuffer> source = [impl_->hardware.device newBufferWithBytes:dispatch.src
                                                          length:field_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> velocity = [impl_->hardware.device newBufferWithBytes:dispatch.velocity
                                                            length:field_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> destination = [impl_->hardware.device newBufferWithBytes:dispatch.src
                                                               length:field_bytes
                                                              options:MTLResourceStorageModeShared];
        if (source == nil || velocity == nil || destination == nil)
            return false;
        AdvectionYConstants constants{};
        constants.nr = dispatch.nr;
        constants.nth = dispatch.nth;
        constants.nz = dispatch.nz;
        constants.dy = dispatch.primary_spacing;
        constants.dt = dispatch.dt;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::AdvectY).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectY)];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:velocity offset:0 atIndex:1];
        [encoder setBuffer:destination offset:0 atIndex:2];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(interior_cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;
        std::memcpy(dispatch.dst, destination.contents, field_bytes);
        ++impl_->telemetry.completed_advection_y_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_vertical_flux(const VerticalFluxDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        if (!supports_vertical_flux_dispatch() || dispatch.q_data == nullptr ||
            dispatch.w_data == nullptr || dispatch.dqdt_data == nullptr || dispatch.dz_data == nullptr ||
            dispatch.dqdt_data == dispatch.q_data || dispatch.dqdt_data == dispatch.w_data ||
            dispatch.dz_count != dispatch.nz || dispatch.limiter_id < 0 || dispatch.limiter_id > 4 ||
            !std::isfinite(dispatch.positivity_dt) ||
            (dispatch.positivity && dispatch.positivity_dt <= 0.0) ||
            !std::isfinite(dispatch.cfl_target) || dispatch.cfl_target < 0.0 ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        std::vector<float> dz(static_cast<std::size_t>(dispatch.nz));
        for (int k = 0; k < dispatch.nz; ++k)
        {
            if (!std::isfinite(dispatch.dz_data[k]) || dispatch.dz_data[k] <= 0.0)
                return false;
            dz[static_cast<std::size_t>(k)] = static_cast<float>(dispatch.dz_data[k]);
            if (!std::isfinite(dz[static_cast<std::size_t>(k)]) || dz[static_cast<std::size_t>(k)] <= 0.0f)
                return false;
        }
        const std::size_t dz_bytes = dz.size() * sizeof(float);
        if (dz_bytes > impl_->max_buffer_bytes())
            return false;

        id<MTLBuffer> q = [impl_->hardware.device newBufferWithBytes:dispatch.q_data
                                                    length:field_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> w = [impl_->hardware.device newBufferWithBytes:dispatch.w_data
                                                    length:field_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> dz_buffer = [impl_->hardware.device newBufferWithBytes:dz.data()
                                                            length:dz_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> tendency = [impl_->hardware.device newBufferWithLength:field_bytes
                                                            options:MTLResourceStorageModeShared];
        if (q == nil || w == nil || dz_buffer == nil || tendency == nil)
            return false;

        TvdVerticalFluxConstants constants{};
        constants.nr = dispatch.nr;
        constants.nth = dispatch.nth;
        constants.nz = dispatch.nz;
        constants.limiter_id = dispatch.limiter_id;
        constants.positivity = dispatch.positivity ? 1 : 0;
        constants.positivity_dt = static_cast<float>(
            std::max(std::abs(dispatch.positivity_dt), 1.0e-12));
        constants.cfl_target = static_cast<float>(dispatch.cfl_target);

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::TvdVerticalFlux).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::TvdVerticalFlux)];
        [encoder setBuffer:q offset:0 atIndex:0];
        [encoder setBuffer:w offset:0 atIndex:1];
        [encoder setBuffer:dz_buffer offset:0 atIndex:2];
        [encoder setBuffer:tendency offset:0 atIndex:3];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(cells, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        double max_cfl = 0.0;
        for (std::size_t index = 0; index < cells; ++index)
        {
            const double speed = std::abs(static_cast<double>(dispatch.w_data[index]));
            if (std::isfinite(speed))
            {
                const std::size_t k = index % static_cast<std::size_t>(dispatch.nz);
                max_cfl = std::max(max_cfl, speed / dispatch.dz_data[k]);
            }
            else if (std::isinf(speed))
            {
                max_cfl = std::numeric_limits<double>::infinity();
            }
        }
        VerticalFluxDispatchResult result;
        result.max_cfl_z = max_cfl;
        result.suggested_dt = max_cfl > 1.0e-30
            ? dispatch.cfl_target / max_cfl
            : 1.0e30;
        std::memcpy(dispatch.dqdt_data, tendency.contents, field_bytes);
        if (dispatch.result != nullptr)
            *dispatch.result = result;
        ++impl_->telemetry.completed_vertical_flux_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_advect_scalars_conservative_cartesian_cgrid(
    const ConservativeCartesianCgridTransportDispatch& dispatch)
{
    @autoreleasepool
    {
        constexpr int kMaximumScalars = 16;
        std::size_t cells = 0, field_bytes = 0;
        if (!supports_conservative_cgrid_transport() || dispatch.u == nullptr ||
            dispatch.v == nullptr || dispatch.w == nullptr || dispatch.rho_data == nullptr ||
            dispatch.q_in == nullptr || dispatch.q_out == nullptr || dispatch.dt_kappas == nullptr ||
            dispatch.n_scalars < 1 || dispatch.n_scalars > kMaximumScalars ||
            dispatch.nx < 3 || dispatch.ny < 3 || dispatch.nz < 1 ||
            !std::isfinite(dispatch.inv_dx) || dispatch.inv_dx <= 0.0f ||
            !std::isfinite(dispatch.inv_dy) || dispatch.inv_dy <= 0.0f ||
            !std::isfinite(dispatch.inv_dz) || dispatch.inv_dz <= 0.0f ||
            !std::isfinite(dispatch.inv_dx2) || dispatch.inv_dx2 <= 0.0f ||
            !std::isfinite(dispatch.inv_dy2) || dispatch.inv_dy2 <= 0.0f ||
            !std::isfinite(dispatch.inv_dz2) || dispatch.inv_dz2 <= 0.0f ||
            !std::isfinite(dispatch.dt_step) || dispatch.dt_step <= 0.0f ||
            !std::isfinite(dispatch.rho_floor) || dispatch.rho_floor <= 0.0f ||
            !std::isfinite(dispatch.frame_velocity_u_ms) ||
            !std::isfinite(dispatch.frame_velocity_v_ms) ||
            !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
            return false;
        if (dispatch.use_supplied_mass_flux &&
            (dispatch.density_begin == nullptr || dispatch.mass_flux_x == nullptr ||
             dispatch.mass_flux_y == nullptr || dispatch.mass_flux_z == nullptr))
            return false;

        std::uint64_t pd_scalars = 0;
        for (int scalar = 0; scalar < dispatch.n_scalars; ++scalar)
        {
            if (dispatch.preserve_positive_support &&
                (!dispatch.positive_definite || dispatch.positive_definite[scalar] == 0))
                return false;
            if (dispatch.q_in[scalar] == nullptr || dispatch.q_out[scalar] == nullptr ||
                !std::isfinite(dispatch.dt_kappas[scalar]) || dispatch.dt_kappas[scalar] != 0.0f)
                return false;
            for (int prior = 0; prior < scalar; ++prior)
            {
                if (dispatch.q_out[scalar] == dispatch.q_out[prior] ||
                    dispatch.q_out[scalar] == dispatch.q_in[prior] ||
                    dispatch.q_out[prior] == dispatch.q_in[scalar])
                    return false;
            }
            if (dispatch.positive_definite != nullptr && dispatch.positive_definite[scalar] != 0)
                ++pd_scalars;
            const auto* energy_report = dispatch.host_energy_reports != nullptr
                ? dispatch.host_energy_reports[scalar] : nullptr;
            const auto energy_kind = dispatch.host_energy_kinds != nullptr
                ? dispatch.host_energy_kinds[scalar] : tmv::ScalarMoistEnergyKind::None;
            if ((energy_report == nullptr) != (energy_kind == tmv::ScalarMoistEnergyKind::None))
                return false;
            const bool report_requested =
                (dispatch.host_boundary_reports != nullptr &&
                 dispatch.host_boundary_reports[scalar] != nullptr) || energy_report != nullptr;
            if (report_requested && !dispatch.use_supplied_mass_flux)
                return false;
        }
        nimbus::compute::CartesianScalarCapture capture;
        if (!capture.prepare(dispatch) || capture.bytes() > impl_->max_buffer_bytes())
            return false;
        id<MTLBuffer> interface_flux = [impl_->hardware.device newBufferWithLength:capture.bytes()
                                                                          options:MTLResourceStorageModeShared];
        if (interface_flux == nil)
            return false;

        static const char* const resident_names[8] = {
            "u", "v", "w", "rho", "cgrid_density_begin",
            "cgrid_mass_flux_x", "cgrid_mass_flux_y", "cgrid_mass_flux_z"};
        const float* resident_sources[8] = {
            dispatch.u, dispatch.v, dispatch.w, dispatch.rho_data,
            dispatch.density_begin, dispatch.mass_flux_x, dispatch.mass_flux_y, dispatch.mass_flux_z};
        const std::size_t resident_count = dispatch.use_supplied_mass_flux ? 8u : 4u;
        std::size_t resident_sizes[8] = {
            field_bytes, field_bytes, field_bytes, field_bytes,
            field_bytes, field_bytes, field_bytes, field_bytes};
        Impl::ResidentAllocation allocations[8];
        if (!impl_->resident.fields.acquire_many(
                resident_names, resident_sizes, resident_count, allocations))
            return false;
        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        Impl::ResidentSlot* resident_slots[8] = {};
        bool reusable[8] = {};
        std::uint64_t resident_reuses = 0;
        std::uint64_t resident_uploads = 0;
        for (std::size_t index = 0; index < resident_count; ++index)
        {
            resident_slots[index] = impl_->resident_slot(allocations[index]);
            if (resident_slots[index] == nullptr || resident_slots[index]->current == nil ||
                resident_slots[index]->current.contents == nullptr || resident_slots[index]->bytes < field_bytes)
                return false;
            reusable[index] = impl_->resident.fields.reusable(
                resident_names[index], resident_sources[index], extent, true);
            if (reusable[index])
                ++resident_reuses;
            else
            {
                impl_->resident.fields.invalidate(resident_names[index]);
                std::memcpy(resident_slots[index]->current.contents, resident_sources[index], field_bytes);
                ++resident_uploads;
            }
        }

        id<MTLBuffer> mfx = dispatch.use_supplied_mass_flux
            ? resident_slots[5]->current
            : [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> mfy = dispatch.use_supplied_mass_flux
            ? resident_slots[6]->current
            : [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> mfz = dispatch.use_supplied_mass_flux
            ? resident_slots[7]->current
            : [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> rho_begin = dispatch.use_supplied_mass_flux
            ? resident_slots[4]->current : resident_slots[3]->current;
        id<MTLBuffer> scalar_inputs[kMaximumScalars] = {};
        id<MTLBuffer> scalar_outputs[kMaximumScalars] = {};
        id<MTLBuffer> requested_outputs[kMaximumScalars] = {};
        if (mfx == nil || mfy == nil || mfz == nil)
            return false;
        for (int scalar = 0; scalar < dispatch.n_scalars; ++scalar)
        {
            scalar_inputs[scalar] = [impl_->hardware.device newBufferWithBytes:dispatch.q_in[scalar]
                                                               length:field_bytes
                                                              options:MTLResourceStorageModeShared];
            scalar_outputs[scalar] = [impl_->hardware.device newBufferWithLength:field_bytes
                                                                 options:MTLResourceStorageModeShared];
            requested_outputs[scalar] = capture.report_index(scalar) >= 0
                ? [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared]
                : interface_flux;
            if (scalar_inputs[scalar] == nil || scalar_outputs[scalar] == nil || requested_outputs[scalar] == nil)
                return false;
        }

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        if (!dispatch.use_supplied_mass_flux)
        {
            id<MTLComputeCommandEncoder> mass_encoder = [command_buffer computeCommandEncoder];
            if (mass_encoder == nil)
                return false;
            AdvectMassFluxConstants constants{};
            constants.nx = dispatch.nx; constants.ny = dispatch.ny; constants.nz = dispatch.nz;
            constants.frame_velocity_u_ms = dispatch.frame_velocity_u_ms;
            constants.frame_velocity_v_ms = dispatch.frame_velocity_v_ms;
            [mass_encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid)];
            [mass_encoder setBuffer:resident_slots[0]->current offset:0 atIndex:0];
            [mass_encoder setBuffer:resident_slots[1]->current offset:0 atIndex:1];
            [mass_encoder setBuffer:resident_slots[2]->current offset:0 atIndex:2];
            [mass_encoder setBuffer:resident_slots[3]->current offset:0 atIndex:3];
            [mass_encoder setBuffer:mfx offset:0 atIndex:4];
            [mass_encoder setBuffer:mfy offset:0 atIndex:5];
            [mass_encoder setBuffer:mfz offset:0 atIndex:6];
            [mass_encoder setBytes:&constants length:sizeof(constants) atIndex:7];
            const NSUInteger maximum = impl_->pipeline(KernelId::AdvectConservativeMassFluxCartesianCgrid).maxTotalThreadsPerThreadgroup;
            if (maximum == 0) { [mass_encoder endEncoding]; return false; }
            [mass_encoder dispatchThreads:MTLSizeMake(cells,1,1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,maximum),1,1)];
            [mass_encoder endEncoding];
        }

        id<MTLComputeCommandEncoder> update_encoder = [command_buffer computeCommandEncoder];
        if (update_encoder == nil)
            return false;
        const NSUInteger update_maximum =
            impl_->pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid).maxTotalThreadsPerThreadgroup;
        if (update_maximum == 0) { [update_encoder endEncoding]; return false; }
        [update_encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectConservativeUpdateCartesianCgrid)];
        [update_encoder setBuffer:resident_slots[3]->current offset:0 atIndex:1];
        [update_encoder setBuffer:mfx offset:0 atIndex:2];
        [update_encoder setBuffer:mfy offset:0 atIndex:3];
        [update_encoder setBuffer:mfz offset:0 atIndex:4];
        [update_encoder setBuffer:rho_begin offset:0 atIndex:5];
        [update_encoder setBuffer:interface_flux offset:0 atIndex:8];
        for (int scalar = 0; scalar < dispatch.n_scalars; ++scalar)
        {
            AdvectUpdateConstants constants{};
            capture.configure(scalar, constants);
            constants.weno  = (dispatch.weno ? 1 : 0) | (dispatch.preserve_positive_support ? 2 : 0);
            constants.nx    = dispatch.nx;
            constants.ny    = dispatch.ny;
            constants.nz    = dispatch.nz;
            constants.use_pd=dispatch.positive_definite != nullptr && dispatch.positive_definite[scalar] != 0;
            constants.inv_dx=dispatch.inv_dx; constants.inv_dy=dispatch.inv_dy; constants.inv_dz=dispatch.inv_dz;
            constants.dt_step=dispatch.dt_step; constants.rho_floor=dispatch.rho_floor;
            constants.use_coupled_mass=dispatch.use_supplied_mass_flux ? 1 : 0;
            [update_encoder setBuffer:scalar_inputs[scalar] offset:0 atIndex:0];
            [update_encoder setBuffer:scalar_outputs[scalar] offset:0 atIndex:6];
            [update_encoder setBuffer:requested_outputs[scalar] offset:0 atIndex:9];
            [update_encoder setBytes:&constants length:sizeof(constants) atIndex:7];
            [update_encoder dispatchThreads:MTLSizeMake(cells,1,1)
                   threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,update_maximum),1,1)];
        }
        [update_encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        nimbus::compute::publish_cartesian_scalar_batch(
            dispatch, capture, static_cast<const float*>(interface_flux.contents), cells,
            [&](int scalar)
            {
                return static_cast<const float*>(scalar_outputs[scalar].contents);
            },
            impl_->scalar_attribution_workspace,
            [&](int scalar)
            {
                return capture.report_index(scalar) >= 0
                    ? static_cast<const float*>(requested_outputs[scalar].contents) : nullptr;
            },
            [](void* destination, const void* source, std::size_t bytes)
            {
                std::memcpy(destination, source, bytes);
            });
        for (std::size_t index = 0; index < resident_count; ++index)
            impl_->resident.fields.commit_host_mirror(resident_names[index], resident_sources[index], extent);
        impl_->telemetry.resident_uploads += resident_uploads;
        impl_->telemetry.resident_reuses += resident_reuses;
        impl_->telemetry.conservative_cgrid_resident_reuses += resident_reuses;
        ++impl_->telemetry.completed_conservative_cgrid_batches;
        if (!dispatch.use_supplied_mass_flux)
            ++impl_->telemetry.completed_conservative_cgrid_mass_flux_dispatches;
        impl_->telemetry.completed_conservative_cgrid_scalars += static_cast<std::uint64_t>(dispatch.n_scalars);
        impl_->telemetry.completed_conservative_cgrid_update_dispatches +=
            static_cast<std::uint64_t>(dispatch.n_scalars);
        impl_->telemetry.completed_conservative_cgrid_pd_scalars += pd_scalars;
        return true;
    }
}

bool MetalComputeBackend::dispatch_advection_batch_pre_vertical(
    const AdvectionBatchPreVerticalDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        std::size_t x_interior_cells = 0;
        std::size_t x_interior_bytes = 0;
        std::size_t y_interior_cells = 0;
        std::size_t y_interior_bytes = 0;
        if (!supports_advection_pre_vertical_batch() || dispatch.scalar_in == nullptr ||
            dispatch.result_out == nullptr || dispatch.u_data == nullptr || dispatch.v_data == nullptr ||
            dispatch.result_out == dispatch.u_data || dispatch.result_out == dispatch.v_data ||
            dispatch.nr < 3 || dispatch.nth < 3 || dispatch.nz < 3 ||
            !std::isfinite(dispatch.dr) || dispatch.dr <= 0.0f ||
            !std::isfinite(dispatch.dtheta) || dispatch.dtheta <= 0.0f ||
            !std::isfinite(dispatch.dt_half) || dispatch.dt_half <= 0.0f ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            !checked_bytes(dispatch.nr - 2, dispatch.nth, dispatch.nz - 2,
                           x_interior_cells, x_interior_bytes) ||
            !checked_bytes(dispatch.nr, dispatch.nth - 2, dispatch.nz - 2,
                           y_interior_cells, y_interior_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        id<MTLBuffer> source = [impl_->hardware.device newBufferWithBytes:dispatch.scalar_in
                                                          length:field_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> u = [impl_->hardware.device newBufferWithBytes:dispatch.u_data
                                                     length:field_bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> v = [impl_->hardware.device newBufferWithBytes:dispatch.v_data
                                                     length:field_bytes
                                                    options:MTLResourceStorageModeShared];
        // Both buffers start with the input so each directional kernel retains
        // ownership of the boundary cells it does not update.
        id<MTLBuffer> after_x = [impl_->hardware.device newBufferWithBytes:dispatch.scalar_in
                                                           length:field_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> after_y = [impl_->hardware.device newBufferWithLength:field_bytes
                                                            options:MTLResourceStorageModeShared];
        if (source == nil || u == nil || v == nil || after_x == nil || after_y == nil)
            return false;

        AdvectionXConstants x_constants{};
        x_constants.nr = dispatch.nr;
        x_constants.nth = dispatch.nth;
        x_constants.nz = dispatch.nz;
        x_constants.dx = dispatch.dr;
        x_constants.dt = dispatch.dt_half;
        AdvectionYConstants y_constants{};
        y_constants.nr = dispatch.nr;
        y_constants.nth = dispatch.nth;
        y_constants.nz = dispatch.nz;
        y_constants.dy = dispatch.dtheta;
        y_constants.dt = dispatch.dt_half;

        const NSUInteger x_maximum = impl_->pipeline(KernelId::AdvectX).maxTotalThreadsPerThreadgroup;
        const NSUInteger y_maximum = impl_->pipeline(KernelId::AdvectY).maxTotalThreadsPerThreadgroup;
        if (x_maximum == 0 || y_maximum == 0)
            return false;

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> x_encoder = [command_buffer computeCommandEncoder];
        if (x_encoder == nil)
            return false;
        [x_encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectX)];
        [x_encoder setBuffer:source offset:0 atIndex:0];
        [x_encoder setBuffer:u offset:0 atIndex:1];
        [x_encoder setBuffer:after_x offset:0 atIndex:2];
        [x_encoder setBytes:&x_constants length:sizeof(x_constants) atIndex:3];
        [x_encoder dispatchThreads:MTLSizeMake(x_interior_cells, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, x_maximum), 1, 1)];
        [x_encoder endEncoding];

        id<MTLBlitCommandEncoder> copy_encoder = [command_buffer blitCommandEncoder];
        if (copy_encoder == nil)
            return false;
        [copy_encoder copyFromBuffer:after_x sourceOffset:0
                            toBuffer:after_y destinationOffset:0 size:field_bytes];
        [copy_encoder endEncoding];

        id<MTLComputeCommandEncoder> y_encoder = [command_buffer computeCommandEncoder];
        if (y_encoder == nil)
            return false;
        [y_encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectY)];
        [y_encoder setBuffer:after_x offset:0 atIndex:0];
        [y_encoder setBuffer:v offset:0 atIndex:1];
        [y_encoder setBuffer:after_y offset:0 atIndex:2];
        [y_encoder setBytes:&y_constants length:sizeof(y_constants) atIndex:3];
        [y_encoder dispatchThreads:MTLSizeMake(y_interior_cells, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, y_maximum), 1, 1)];
        [y_encoder endEncoding];

        if (!impl_->finish(command_buffer))
            return false;
        std::memcpy(dispatch.result_out, after_y.contents, field_bytes);
        ++impl_->telemetry.completed_advection_x_dispatches;
        ++impl_->telemetry.completed_advection_y_dispatches;
        ++impl_->telemetry.completed_advection_pre_vertical_batches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_advection_batch_post_vertical(
    const AdvectionBatchPostVerticalDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells = 0;
        std::size_t field_bytes = 0;
        std::size_t y_interior_cells = 0;
        std::size_t y_interior_bytes = 0;
        std::size_t x_interior_cells = 0;
        std::size_t x_interior_bytes = 0;
        if (!supports_advection_pre_vertical_batch() || dispatch.scalar_in == nullptr ||
            dispatch.result_out == nullptr || dispatch.u_data == nullptr || dispatch.v_data == nullptr ||
            dispatch.result_out == dispatch.u_data || dispatch.result_out == dispatch.v_data ||
            dispatch.nr < 3 || dispatch.nth < 3 || dispatch.nz < 3 ||
            !std::isfinite(dispatch.dr) || dispatch.dr <= 0.0f ||
            !std::isfinite(dispatch.dtheta) || dispatch.dtheta <= 0.0f ||
            !std::isfinite(dispatch.dz) || dispatch.dz <= 0.0f ||
            !std::isfinite(dispatch.dt_half) || dispatch.dt_half <= 0.0f ||
            !std::isfinite(dispatch.dt_full) || dispatch.dt_full <= 0.0f ||
            !std::isfinite(dispatch.kappa) || dispatch.kappa != 0.0f ||
            !checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, field_bytes) ||
            !checked_bytes(dispatch.nr, dispatch.nth - 2, dispatch.nz - 2,
                           y_interior_cells, y_interior_bytes) ||
            !checked_bytes(dispatch.nr - 2, dispatch.nth, dispatch.nz - 2,
                           x_interior_cells, x_interior_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        id<MTLBuffer> source = [impl_->hardware.device newBufferWithBytes:dispatch.scalar_in
                                                          length:field_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> u = [impl_->hardware.device newBufferWithBytes:dispatch.u_data
                                                     length:field_bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> v = [impl_->hardware.device newBufferWithBytes:dispatch.v_data
                                                     length:field_bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> after_y = [impl_->hardware.device newBufferWithBytes:dispatch.scalar_in
                                                           length:field_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> after_x = [impl_->hardware.device newBufferWithLength:field_bytes
                                                            options:MTLResourceStorageModeShared];
        if (source == nil || u == nil || v == nil || after_y == nil || after_x == nil)
            return false;

        AdvectionYConstants y_constants{};
        y_constants.nr = dispatch.nr;
        y_constants.nth = dispatch.nth;
        y_constants.nz = dispatch.nz;
        y_constants.dy = dispatch.dtheta;
        y_constants.dt = dispatch.dt_half;
        AdvectionXConstants x_constants{};
        x_constants.nr = dispatch.nr;
        x_constants.nth = dispatch.nth;
        x_constants.nz = dispatch.nz;
        x_constants.dx = dispatch.dr;
        x_constants.dt = dispatch.dt_half;

        const NSUInteger y_maximum = impl_->pipeline(KernelId::AdvectY).maxTotalThreadsPerThreadgroup;
        const NSUInteger x_maximum = impl_->pipeline(KernelId::AdvectX).maxTotalThreadsPerThreadgroup;
        if (y_maximum == 0 || x_maximum == 0)
            return false;
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;

        id<MTLComputeCommandEncoder> y_encoder = [command_buffer computeCommandEncoder];
        if (y_encoder == nil)
            return false;
        [y_encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectY)];
        [y_encoder setBuffer:source offset:0 atIndex:0];
        [y_encoder setBuffer:v offset:0 atIndex:1];
        [y_encoder setBuffer:after_y offset:0 atIndex:2];
        [y_encoder setBytes:&y_constants length:sizeof(y_constants) atIndex:3];
        [y_encoder dispatchThreads:MTLSizeMake(y_interior_cells, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, y_maximum), 1, 1)];
        [y_encoder endEncoding];

        id<MTLBlitCommandEncoder> copy_encoder = [command_buffer blitCommandEncoder];
        if (copy_encoder == nil)
            return false;
        [copy_encoder copyFromBuffer:after_y sourceOffset:0
                            toBuffer:after_x destinationOffset:0 size:field_bytes];
        [copy_encoder endEncoding];

        id<MTLComputeCommandEncoder> x_encoder = [command_buffer computeCommandEncoder];
        if (x_encoder == nil)
            return false;
        [x_encoder setComputePipelineState:impl_->pipeline(KernelId::AdvectX)];
        [x_encoder setBuffer:after_y offset:0 atIndex:0];
        [x_encoder setBuffer:u offset:0 atIndex:1];
        [x_encoder setBuffer:after_x offset:0 atIndex:2];
        [x_encoder setBytes:&x_constants length:sizeof(x_constants) atIndex:3];
        [x_encoder dispatchThreads:MTLSizeMake(x_interior_cells, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, x_maximum), 1, 1)];
        [x_encoder endEncoding];

        if (!impl_->finish(command_buffer))
            return false;
        std::memcpy(dispatch.result_out, after_x.contents, field_bytes);
        ++impl_->telemetry.completed_advection_y_dispatches;
        ++impl_->telemetry.completed_advection_x_dispatches;
        ++impl_->telemetry.completed_advection_post_vertical_batches;
        return true;
    }
}
