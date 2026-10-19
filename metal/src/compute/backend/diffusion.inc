// Explicit Cartesian diffusion and atomic batch execution.
DiffusionComputeKernels* MetalComputeBackend::diffusion_kernels()
{
    return supports_diffusion_dispatch() ? this : nullptr;
}

const DiffusionComputeKernels* MetalComputeBackend::diffusion_kernels() const
{
    return supports_diffusion_dispatch() ? this : nullptr;
}

bool MetalComputeBackend::supports(DiffusionKernelOp op) const
{
    if (op == DiffusionKernelOp::Implicit)
        return ready(KernelId::ImplicitDiffusionStencil) && ready(KernelId::ImplicitDiffusionVector);
    return op == DiffusionKernelOp::Diffusion && supports_diffusion_dispatch();
}

bool MetalComputeBackend::supports_diffusion_dispatch() const noexcept
{
    return ready(nimbus::compute::KernelId::Diffusion);
}

bool MetalComputeBackend::dispatch_diffusion(const DiffusionDispatch& dispatch)
{
    const float*                 fields[1]     = {dispatch.field};
    float*                       tendencies[1] = {dispatch.tendency};
    const DiffusionFieldLocation locations[1]  = {dispatch.location};
    const DiffusionBatchDispatch batch{fields,
                                       tendencies,
                                       locations,
                                       1,
                                       dispatch.rho,
                                       dispatch.dz,
                                       dispatch.nx,
                                       dispatch.ny,
                                       dispatch.nz,
                                       dispatch.dx,
                                       dispatch.dy,
                                       dispatch.K_h,
                                       dispatch.K_v,
                                       dispatch.c_grid,
                                       dispatch.periodic_x,
                                       dispatch.periodic_y,
                                       dispatch.vertical_K == nullptr ? nullptr : &dispatch.vertical_K};
    const bool                   completed = dispatch_diffusion_batch(batch);
    if (completed)
        --impl_->telemetry.completed_diffusion_batches;
    return completed;
}

bool MetalComputeBackend::dispatch_diffusion_batch(const DiffusionBatchDispatch& dispatch)
{
    @autoreleasepool
    {
        constexpr int kMaximumTargets = 5;
        std::size_t   cells           = 0;
        std::size_t   field_bytes     = 0;
        if (!supports_diffusion_dispatch() || dispatch.fields == nullptr || dispatch.tendencies == nullptr ||
            dispatch.locations == nullptr || dispatch.rho == nullptr || dispatch.dz == nullptr || dispatch.count < 1 ||
            dispatch.count > kMaximumTargets || dispatch.nx < 3 || dispatch.ny < 3 || dispatch.nz < 2 ||
            !std::isfinite(dispatch.dx) || dispatch.dx <= 0.0f || !std::isfinite(dispatch.dy) || dispatch.dy <= 0.0f ||
            !std::isfinite(dispatch.K_h) || dispatch.K_h < 0.0f || !std::isfinite(dispatch.K_v) ||
            dispatch.K_v < 0.0f || !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }
        for (int target = 0; target < dispatch.count; ++target)
        {
            const int location = static_cast<int>(dispatch.locations[target]);
            if (dispatch.fields[target] == nullptr || dispatch.tendencies[target] == nullptr ||
                location < static_cast<int>(DiffusionFieldLocation::Scalar) ||
                location > static_cast<int>(DiffusionFieldLocation::WFace))
                return false;
            for (int prior = 0; prior < target; ++prior)
                if (dispatch.tendencies[target] == dispatch.tendencies[prior])
                    return false;
            if (dispatch.tendencies[target] == dispatch.rho || dispatch.tendencies[target] == dispatch.dz)
                return false;
            for (int input = 0; input < dispatch.count; ++input)
                if (dispatch.tendencies[target] == dispatch.fields[input])
                    return false;
        }
        for (int k = 0; k < dispatch.nz; ++k)
            if (!std::isfinite(dispatch.dz[k]) || dispatch.dz[k] <= 0.0f)
                return false;
        for (std::size_t index = 0; index < cells; ++index)
        {
            if (!std::isfinite(dispatch.rho[index]) || dispatch.rho[index] <= 0.0f)
                return false;
            for (int target = 0; target < dispatch.count; ++target)
                if (!std::isfinite(dispatch.fields[target][index]))
                    return false;
            if (dispatch.vertical_K != nullptr)
                for (int target = 0; target < dispatch.count; ++target)
                    if (dispatch.vertical_K[target] == nullptr || !std::isfinite(dispatch.vertical_K[target][index]) ||
                        dispatch.vertical_K[target][index] < 0.0f)
                        return false;
        }

        const std::size_t dz_bytes = static_cast<std::size_t>(dispatch.nz) * sizeof(float);
        id<MTLBuffer>     rho      = [impl_->hardware.device newBufferWithBytes:dispatch.rho
                                                                         length:field_bytes
                                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer>     dz       = [impl_->hardware.device newBufferWithBytes:dispatch.dz
                                                                         length:dz_bytes
                                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer>     fields[kMaximumTargets]     = {};
        id<MTLBuffer>     vertical_k[kMaximumTargets] = {};
        id<MTLBuffer>     tendencies[kMaximumTargets] = {};
        if (rho == nil || dz == nil)
            return false;
        for (int target = 0; target < dispatch.count; ++target)
        {
            fields[target] = [impl_->hardware.device newBufferWithBytes:dispatch.fields[target]
                                                                 length:field_bytes
                                                                options:MTLResourceStorageModeShared];
            tendencies[target] =
                [impl_->hardware.device newBufferWithLength:field_bytes options:MTLResourceStorageModeShared];
            if (dispatch.vertical_K != nullptr)
                vertical_k[target] = [impl_->hardware.device newBufferWithBytes:dispatch.vertical_K[target]
                                                                         length:field_bytes
                                                                        options:MTLResourceStorageModeShared];
            if (fields[target] == nil || tendencies[target] == nil ||
                (dispatch.vertical_K != nullptr && vertical_k[target] == nil))
                return false;
        }

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;
        const NSUInteger maximum = impl_->pipeline(KernelId::Diffusion).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:impl_->pipeline(KernelId::Diffusion)];
        [encoder setBuffer:rho offset:0 atIndex:1];
        [encoder setBuffer:dz offset:0 atIndex:2];
        const NSUInteger width = std::min<NSUInteger>(64, maximum);
        for (int target = 0; target < dispatch.count; ++target)
        {
            DiffusionConstants constants{};
            constants.nx         = dispatch.nx;
            constants.ny         = dispatch.ny;
            constants.nz         = dispatch.nz;
            constants.location   = static_cast<std::int32_t>(dispatch.locations[target]);
            constants.c_grid     = dispatch.c_grid ? 1 : 0;
            constants.periodic_x = dispatch.periodic_x ? 1 : 0;
            constants.periodic_y = dispatch.periodic_y ? 1 : 0;
            constants.padding_i  = dispatch.vertical_K == nullptr ? 0 : 1;
            constants.inv_dx2    = 1.0f / (dispatch.dx * dispatch.dx);
            constants.inv_dy2    = 1.0f / (dispatch.dy * dispatch.dy);
            constants.K_h        = dispatch.K_h;
            constants.K_v        = dispatch.K_v;
            [encoder setBuffer:fields[target] offset:0 atIndex:0];
            [encoder setBuffer:tendencies[target] offset:0 atIndex:3];
            [encoder setBuffer:dispatch.vertical_K == nullptr ? rho : vertical_k[target] offset:0 atIndex:4];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:5];
            [encoder dispatchThreads:MTLSizeMake(cells, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        }
        [encoder endEncoding];
        if (!impl_->finish(command_buffer))
            return false;

        for (int target = 0; target < dispatch.count; ++target)
            std::memcpy(dispatch.tendencies[target], tendencies[target].contents, field_bytes);
        impl_->telemetry.completed_diffusion_dispatches += static_cast<std::uint64_t>(dispatch.count);
        ++impl_->telemetry.completed_diffusion_batches;
        return true;
    }
}

namespace
{
template <class Implementation, class Constants, std::size_t Inputs>
bool dispatch_implicit_fields_metal(Implementation&                         impl,
                                    KernelId                                kernel,
                                    const Constants&                        constants,
                                    const std::array<const float*, Inputs>& inputs,
                                    const std::array<std::size_t, Inputs>&  input_sizes,
                                    const std::array<float*, 5>&            outputs,
                                    const std::array<std::size_t, 5>&       output_sizes,
                                    std::size_t                             columns)
{
    id<MTLBuffer> buffers[Inputs + 5];
    for (std::size_t i = 0; i < Inputs; ++i)
    {
        if (input_sizes[i] * sizeof(float) > impl.max_buffer_bytes())
            return false;
        buffers[i] = [impl.hardware.device newBufferWithBytes:inputs[i]
                                                       length:input_sizes[i] * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    }
    for (std::size_t i = 0; i < 5; ++i)
    {
        if (output_sizes[i] * sizeof(float) > impl.max_buffer_bytes())
            return false;
        buffers[Inputs + i] = [impl.hardware.device newBufferWithLength:output_sizes[i] * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
    }
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
    for (NSUInteger i = 0; i < Inputs + 5; ++i)
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder setBytes:&constants length:sizeof(constants) atIndex:Inputs + 5];
    [encoder dispatchThreads:MTLSizeMake(columns, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    if (!impl.finish(command))
        return false;
    if (kernel == KernelId::ImplicitDiffusionStencil)
        ++impl.telemetry.completed_implicit_stencil_dispatches;
    else
        ++impl.telemetry.completed_implicit_vector_dispatches;
    for (std::size_t f = 0; f < 5; ++f)
    {
        auto data = static_cast<const float*>(buffers[Inputs + f].contents);
        if (data == nullptr)
            return false;
        for (std::size_t n = 0; n < output_sizes[f]; ++n)
            if (!std::isfinite(data[n]) || (f >= 3 && data[n] < 0) ||
                (kernel == KernelId::ImplicitDiffusionStencil && f == 1 && data[n] < 0))
                return false;
    }
    for (std::size_t f = 0; f < 5; ++f)
        std::memcpy(outputs[f], buffers[Inputs + f].contents, output_sizes[f] * sizeof(float));
    return true;
}
} // namespace

bool MetalComputeBackend::dispatch_implicit_stencil(const ImplicitDiffusionStencilDispatch& d)
{
    @autoreleasepool
    {
        std::size_t n = 0, c = 0;
        if (!supports(DiffusionKernelOp::Implicit) || !nimbus::compute::implicit_contract::validate(d, n, c))
            return false;
        const auto& a = d.diffusion;
        return dispatch_implicit_fields_metal(
            *impl_,
            KernelId::ImplicitDiffusionStencil,
            nimbus::compute::implicit_contract::constants(d),
            std::array<const float*, 4>{a.field, a.rho, a.dz, a.vertical_K ? a.vertical_K : a.rho},
            std::array<std::size_t, 4>{n, n, static_cast<std::size_t>(a.nz), n},
            std::array<float*, 5>{a.tendency, d.diagonal, d.dot, d.preconditioned_norm, d.norm},
            std::array<std::size_t, 5>{n, n, c, c, c},
            c);
    }
}

bool MetalComputeBackend::dispatch_implicit_vector(const ImplicitDiffusionVectorDispatch& d)
{
    @autoreleasepool
    {
        std::size_t n = 0, c = 0;
        if (!supports(DiffusionKernelOp::Implicit) || !nimbus::compute::implicit_contract::validate(d, n, c))
            return false;
        return dispatch_implicit_fields_metal(
            *impl_,
            KernelId::ImplicitDiffusionVector,
            nimbus::compute::implicit_contract::constants(d),
            std::array<const float*, 7>{d.original, d.rhs, d.diagonal, d.solution, d.residual, d.direction, d.image},
            std::array<std::size_t, 7>{n, n, n, n, n, n, n},
            std::array<float*, 5>{d.next_solution, d.next_residual, d.next_direction, d.preconditioned_norm, d.norm},
            std::array<std::size_t, 5>{n, n, n, c, c},
            c);
    }
}
