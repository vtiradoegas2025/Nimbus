RadiationComputeKernels* MetalComputeBackend::radiation_kernels()
{
    return ready(KernelId::RrtmgLongwaveTransfer) || ready(KernelId::RrtmgShortwaveTransfer)
               ? this : nullptr;
}

const RadiationComputeKernels* MetalComputeBackend::radiation_kernels() const
{
    return ready(KernelId::RrtmgLongwaveTransfer) || ready(KernelId::RrtmgShortwaveTransfer)
               ? this : nullptr;
}

bool MetalComputeBackend::supports(RadiationKernelOp op) const
{
    switch (op)
    {
        case RadiationKernelOp::SimpleGrey:
            return ready(KernelId::SimpleGrey);
        case RadiationKernelOp::RrtmgLongwaveTransfer:
            return ready(KernelId::RrtmgLongwaveTransfer);
        case RadiationKernelOp::RrtmgShortwaveTransfer:
            return ready(KernelId::RrtmgShortwaveTransfer);
        case RadiationKernelOp::RrtmgLongwaveTransferBatch:
            return ready(KernelId::RrtmgLongwaveTransferBatch);
        case RadiationKernelOp::RrtmgShortwaveTransferBatch:
            return ready(KernelId::RrtmgShortwaveTransferBatch);
    }
    return false;
}

bool MetalComputeBackend::dispatch_rrtmg_longwave_transfer_batch(
    const RrtmgLongwaveTransferBatchDispatch& dispatch)
{
    @autoreleasepool
    {
        constexpr int g_points = 140;
        constexpr int bands = 16;
        std::size_t spectral_count = 0, spectral_bytes = 0;
        std::size_t partial_count = 0, partial_bytes = 0;
        std::size_t layer_band_count = 0, layer_band_bytes = 0;
        std::size_t interface_band_count = 0, interface_band_bytes = 0;
        if (!supports(RadiationKernelOp::RrtmgLongwaveTransferBatch) ||
            dispatch.columns <= 0 || dispatch.layers <= 0 ||
            dispatch.layers == std::numeric_limits<int>::max() ||
            dispatch.absorption_optical_depth == nullptr || dispatch.planck_fraction == nullptr ||
            dispatch.layer_planck == nullptr || dispatch.interface_planck == nullptr ||
            dispatch.surface_planck == nullptr || dispatch.band_width_cm == nullptr ||
            dispatch.surface_emissivity == nullptr || dispatch.precipitable_water_cm == nullptr ||
            dispatch.upward_wm2 == nullptr || dispatch.downward_wm2 == nullptr ||
            dispatch.upward_wm2 == dispatch.downward_wm2 ||
            !checked_bytes(dispatch.columns, dispatch.layers, g_points, spectral_count, spectral_bytes) ||
            !checked_bytes(dispatch.columns, dispatch.layers + 1, g_points, partial_count, partial_bytes) ||
            !checked_bytes(dispatch.columns, dispatch.layers, bands, layer_band_count, layer_band_bytes) ||
            !checked_bytes(dispatch.columns, dispatch.layers + 1, bands,
                           interface_band_count, interface_band_bytes) ||
            partial_bytes > impl_->max_buffer_bytes())
            return false;
        const std::size_t columns = static_cast<std::size_t>(dispatch.columns);
        const std::size_t layers = static_cast<std::size_t>(dispatch.layers);
        const std::size_t result_count = columns * (layers + 1);
        const auto valid_nonnegative = [](const float* values, std::size_t count)
        {
            for (std::size_t index = 0; index < count; ++index)
                if (!std::isfinite(values[index]) || values[index] < 0.0f) return false;
            return true;
        };
        if (!valid_nonnegative(dispatch.absorption_optical_depth, spectral_count) ||
            !valid_nonnegative(dispatch.planck_fraction, spectral_count) ||
            !valid_nonnegative(dispatch.layer_planck, layer_band_count) ||
            !valid_nonnegative(dispatch.interface_planck, interface_band_count) ||
            !valid_nonnegative(dispatch.surface_planck, columns * bands) ||
            !valid_nonnegative(dispatch.band_width_cm, bands) ||
            (dispatch.cloud_absorption_optical_depth != nullptr &&
             !valid_nonnegative(dispatch.cloud_absorption_optical_depth, layer_band_count)))
            return false;
        for (int band = 0; band < bands; ++band)
            if (dispatch.band_width_cm[band] <= 0.0f) return false;
        try
        {
            std::vector<RrtmgLongwaveColumnParameters> parameters(columns);
            for (std::size_t column = 0; column < columns; ++column)
            {
                const float emissivity = dispatch.surface_emissivity[column];
                const float water = dispatch.precipitable_water_cm[column];
                if (!std::isfinite(emissivity) || emissivity < 0.0f || emissivity > 1.0f ||
                    !std::isfinite(water) || water < 0.0f) return false;
                parameters[column].surface_emissivity = emissivity;
                parameters[column].precipitable_water_cm = water;
            }
            std::vector<float> zero_clouds;
            const float* cloud = dispatch.cloud_absorption_optical_depth;
            if (cloud == nullptr)
            {
                zero_clouds.assign(layer_band_count, 0.0f);
                cloud = zero_clouds.data();
            }
            const void* inputs[8] = {
                dispatch.absorption_optical_depth, dispatch.planck_fraction,
                dispatch.layer_planck, dispatch.interface_planck, dispatch.surface_planck,
                dispatch.band_width_cm, cloud, parameters.data()};
            const std::size_t input_bytes[8] = {
                spectral_bytes, spectral_bytes, layer_band_bytes, interface_band_bytes,
                columns * bands * sizeof(float), bands * sizeof(float), layer_band_bytes,
                columns * sizeof(RrtmgLongwaveColumnParameters)};
            id<MTLBuffer> input_buffers[8];
            for (NSUInteger index = 0; index < 8; ++index)
            {
                input_buffers[index] = [impl_->hardware.device newBufferWithBytes:inputs[index]
                                                                    length:input_bytes[index]
                                                                   options:MTLResourceStorageModeShared];
                if (input_buffers[index] == nil) return false;
            }
            id<MTLBuffer> partial_up = [impl_->hardware.device newBufferWithLength:partial_bytes
                                                                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> partial_down = [impl_->hardware.device newBufferWithLength:partial_bytes
                                                                              options:MTLResourceStorageModeShared];
            if (partial_up == nil || partial_down == nil) return false;
            id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
            if (command_buffer == nil) return false;
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) return false;
            RrtmgLongwaveTransferBatchConstants constants{};
            constants.layers = dispatch.layers; constants.columns = dispatch.columns;
            constants.has_clouds = dispatch.cloud_absorption_optical_depth != nullptr ? 1 : 0;
            [encoder setComputePipelineState:impl_->pipeline(KernelId::RrtmgLongwaveTransferBatch)];
            for (NSUInteger index = 0; index < 8; ++index)
                [encoder setBuffer:input_buffers[index] offset:0 atIndex:index];
            [encoder setBuffer:partial_up offset:0 atIndex:8];
            [encoder setBuffer:partial_down offset:0 atIndex:9];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:10];
            const NSUInteger maximum = impl_->pipeline(KernelId::RrtmgLongwaveTransferBatch)
                                           .maxTotalThreadsPerThreadgroup;
            if (maximum == 0) { [encoder endEncoding]; return false; }
            [encoder dispatchThreads:MTLSizeMake(columns * g_points, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
            [encoder endEncoding];
            if (!impl_->finish(command_buffer)) return false;
            const float* source_up = static_cast<const float*>(partial_up.contents);
            const float* source_down = static_cast<const float*>(partial_down.contents);
            std::vector<float> upward(result_count, 0.0f), downward(result_count, 0.0f);
            for (std::size_t column = 0; column < columns; ++column)
                for (std::size_t g = 0; g < static_cast<std::size_t>(g_points); ++g)
                    for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                    {
                        const std::size_t partial =
                            (column * g_points + g) * (layers + 1) + interface_index;
                        const std::size_t result = column * (layers + 1) + interface_index;
                        upward[result] += source_up[partial]; downward[result] += source_down[partial];
                    }
            for (std::size_t index = 0; index < result_count; ++index)
                if (!std::isfinite(upward[index]) || upward[index] < 0.0f ||
                    !std::isfinite(downward[index]) || downward[index] < 0.0f) return false;
            std::memcpy(dispatch.upward_wm2, upward.data(), result_count * sizeof(float));
            std::memcpy(dispatch.downward_wm2, downward.data(), result_count * sizeof(float));
            ++impl_->telemetry.completed_rrtmg_longwave_transfer_batch_dispatches;
            return true;
        }
        catch (const std::bad_alloc&) { return false; }
        catch (const std::length_error&) { return false; }
    }
}

bool MetalComputeBackend::dispatch_rrtmg_shortwave_transfer_batch(
    const RrtmgShortwaveTransferBatchDispatch& dispatch)
{
    @autoreleasepool
    {
        constexpr int g_points = 112;
        constexpr int bands = 14;
        const bool any_cloud = dispatch.cloud_optical_depth != nullptr ||
            dispatch.cloud_single_scattering_albedo != nullptr || dispatch.cloud_asymmetry != nullptr;
        const bool all_cloud = dispatch.cloud_optical_depth != nullptr &&
            dispatch.cloud_single_scattering_albedo != nullptr && dispatch.cloud_asymmetry != nullptr;
        std::size_t spectral_count = 0, spectral_bytes = 0, partial_count = 0, partial_bytes = 0;
        std::size_t cloud_count = 0, cloud_bytes = 0;
        if (!supports(RadiationKernelOp::RrtmgShortwaveTransferBatch) ||
            dispatch.columns <= 0 || dispatch.layers <= 0 ||
            dispatch.layers == std::numeric_limits<int>::max() || any_cloud != all_cloud ||
            dispatch.absorption_optical_depth == nullptr || dispatch.rayleigh_optical_depth == nullptr ||
            dispatch.solar_source_fraction == nullptr || dispatch.cosine_zenith == nullptr ||
            dispatch.solar_irradiance_wm2 == nullptr || dispatch.surface_albedo == nullptr ||
            dispatch.upward_wm2 == nullptr || dispatch.downward_wm2 == nullptr ||
            dispatch.upward_wm2 == dispatch.downward_wm2 ||
            !checked_bytes(dispatch.columns, dispatch.layers, g_points, spectral_count, spectral_bytes) ||
            !checked_bytes(dispatch.columns, dispatch.layers + 1, g_points, partial_count, partial_bytes) ||
            !checked_bytes(dispatch.columns, dispatch.layers, bands, cloud_count, cloud_bytes) ||
            partial_bytes > impl_->max_buffer_bytes()) return false;
        const std::size_t columns = static_cast<std::size_t>(dispatch.columns);
        const std::size_t layers = static_cast<std::size_t>(dispatch.layers);
        const std::size_t result_count = columns * (layers + 1);
        const auto valid_range = [](const float* values, std::size_t count, float maximum)
        {
            for (std::size_t index = 0; index < count; ++index)
                if (!std::isfinite(values[index]) || values[index] < 0.0f || values[index] > maximum)
                    return false;
            return true;
        };
        if (!valid_range(dispatch.absorption_optical_depth, spectral_count, std::numeric_limits<float>::max()) ||
            !valid_range(dispatch.rayleigh_optical_depth, spectral_count, std::numeric_limits<float>::max()) ||
            !valid_range(dispatch.solar_source_fraction, columns * g_points, std::numeric_limits<float>::max()) ||
            (all_cloud && (!valid_range(dispatch.cloud_optical_depth, cloud_count, std::numeric_limits<float>::max()) ||
                           !valid_range(dispatch.cloud_single_scattering_albedo, cloud_count, 1.0f) ||
                           !valid_range(dispatch.cloud_asymmetry, cloud_count, 1.0f)))) return false;
        try
        {
            std::vector<RrtmgShortwaveColumnParameters> parameters(columns);
            for (std::size_t column = 0; column < columns; ++column)
            {
                const float mu = dispatch.cosine_zenith[column];
                const float irradiance = dispatch.solar_irradiance_wm2[column];
                const float albedo = dispatch.surface_albedo[column];
                if (!std::isfinite(mu) || mu < 0.0f || mu > 1.0f ||
                    !std::isfinite(irradiance) || irradiance < 0.0f ||
                    !std::isfinite(albedo) || albedo < 0.0f || albedo > 1.0f) return false;
                float sum = 0.0f;
                for (int g = 0; g < g_points; ++g)
                    sum += dispatch.solar_source_fraction[column * g_points + static_cast<std::size_t>(g)];
                if ((!std::isfinite(sum) || sum <= 0.0f) && mu != 0.0f && irradiance != 0.0f) return false;
                parameters[column] = {mu, irradiance, albedo,
                                      std::isfinite(sum) && sum > 0.0f ? sum : 1.0f};
            }
            std::vector<float> zero_clouds;
            const float* clouds[3] = {dispatch.cloud_optical_depth,
                                      dispatch.cloud_single_scattering_albedo, dispatch.cloud_asymmetry};
            if (!all_cloud)
            {
                zero_clouds.assign(cloud_count, 0.0f);
                for (const float*& cloud : clouds) cloud = zero_clouds.data();
            }
            const void* inputs[7] = {dispatch.absorption_optical_depth, dispatch.rayleigh_optical_depth,
                                     dispatch.solar_source_fraction, clouds[0], clouds[1], clouds[2],
                                     parameters.data()};
            const std::size_t input_bytes[7] = {spectral_bytes, spectral_bytes,
                columns * g_points * sizeof(float), cloud_bytes, cloud_bytes, cloud_bytes,
                columns * sizeof(RrtmgShortwaveColumnParameters)};
            id<MTLBuffer> input_buffers[7];
            for (NSUInteger index = 0; index < 7; ++index)
            {
                input_buffers[index] = [impl_->hardware.device newBufferWithBytes:inputs[index]
                                                                    length:input_bytes[index]
                                                                   options:MTLResourceStorageModeShared];
                if (input_buffers[index] == nil) return false;
            }
            id<MTLBuffer> output_buffers[12];
            for (NSUInteger index = 0; index < 12; ++index)
            {
                output_buffers[index] = [impl_->hardware.device newBufferWithLength:partial_bytes
                                                                             options:MTLResourceStorageModeShared];
                if (output_buffers[index] == nil) return false;
            }
            id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
            if (command_buffer == nil) return false;
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) return false;
            RrtmgShortwaveTransferBatchConstants constants{};
            constants.layers = dispatch.layers; constants.columns = dispatch.columns;
            constants.has_clouds = all_cloud ? 1 : 0;
            [encoder setComputePipelineState:impl_->pipeline(KernelId::RrtmgShortwaveTransferBatch)];
            for (NSUInteger index = 0; index < 7; ++index)
                [encoder setBuffer:input_buffers[index] offset:0 atIndex:index];
            for (NSUInteger index = 0; index < 12; ++index)
                [encoder setBuffer:output_buffers[index] offset:0 atIndex:7 + index];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:19];
            const NSUInteger maximum = impl_->pipeline(KernelId::RrtmgShortwaveTransferBatch)
                                           .maxTotalThreadsPerThreadgroup;
            if (maximum == 0) { [encoder endEncoding]; return false; }
            [encoder dispatchThreads:MTLSizeMake(columns * g_points, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, maximum), 1, 1)];
            [encoder endEncoding];
            if (!impl_->finish(command_buffer)) return false;
            const float* source_up = static_cast<const float*>(output_buffers[0].contents);
            const float* source_down = static_cast<const float*>(output_buffers[1].contents);
            std::vector<float> upward(result_count, 0.0f), downward(result_count, 0.0f);
            for (std::size_t column = 0; column < columns; ++column)
                for (std::size_t g = 0; g < static_cast<std::size_t>(g_points); ++g)
                    for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                    {
                        const std::size_t partial =
                            (column * g_points + g) * (layers + 1) + interface_index;
                        const std::size_t result = column * (layers + 1) + interface_index;
                        upward[result] += source_up[partial]; downward[result] += source_down[partial];
                    }
            for (std::size_t index = 0; index < result_count; ++index)
                if (!std::isfinite(upward[index]) || upward[index] < 0.0f ||
                    !std::isfinite(downward[index]) || downward[index] < 0.0f) return false;
            std::memcpy(dispatch.upward_wm2, upward.data(), result_count * sizeof(float));
            std::memcpy(dispatch.downward_wm2, downward.data(), result_count * sizeof(float));
            ++impl_->telemetry.completed_rrtmg_shortwave_transfer_batch_dispatches;
            return true;
        }
        catch (const std::bad_alloc&) { return false; }
        catch (const std::length_error&) { return false; }
    }
}

bool MetalComputeBackend::dispatch_rrtmg_shortwave_transfer(
    const RrtmgShortwaveTransferDispatch& dispatch)
{
    @autoreleasepool
    {
        constexpr std::size_t g_points = 112;
        constexpr std::size_t bands = 14;
        const bool any_cloud = dispatch.cloud_optical_depth != nullptr ||
                               dispatch.cloud_single_scattering_albedo != nullptr ||
                               dispatch.cloud_asymmetry != nullptr;
        const bool all_cloud = dispatch.cloud_optical_depth != nullptr &&
                               dispatch.cloud_single_scattering_albedo != nullptr &&
                               dispatch.cloud_asymmetry != nullptr;
        if (!supports(RadiationKernelOp::RrtmgShortwaveTransfer) ||
            dispatch.layers <= 0 || dispatch.layers == std::numeric_limits<int>::max() ||
            dispatch.absorption_optical_depth == nullptr ||
            dispatch.rayleigh_optical_depth == nullptr ||
            dispatch.solar_source_fraction == nullptr || dispatch.upward_wm2 == nullptr ||
            dispatch.downward_wm2 == nullptr || dispatch.upward_wm2 == dispatch.downward_wm2 ||
            any_cloud != all_cloud || !std::isfinite(dispatch.cosine_zenith) ||
            dispatch.cosine_zenith < 0.0f || dispatch.cosine_zenith > 1.0f ||
            !std::isfinite(dispatch.solar_irradiance_wm2) ||
            dispatch.solar_irradiance_wm2 < 0.0f ||
            !std::isfinite(dispatch.surface_albedo) || dispatch.surface_albedo < 0.0f ||
            dispatch.surface_albedo > 1.0f)
            return false;

        const std::size_t layers = static_cast<std::size_t>(dispatch.layers);
        if (layers > (std::numeric_limits<std::size_t>::max() / g_points) - 1 ||
            layers > (std::numeric_limits<std::size_t>::max() / bands) - 1)
            return false;
        const std::size_t spectral_count = layers * g_points;
        const std::size_t cloud_count = layers * bands;
        const std::size_t partial_count = (layers + 1) * g_points;
        if (partial_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) return false;
        const std::size_t partial_bytes = partial_count * sizeof(float);
        if (partial_bytes > impl_->max_buffer_bytes()) return false;

        const void* inputs[6] = {
            dispatch.absorption_optical_depth, dispatch.rayleigh_optical_depth,
            dispatch.solar_source_fraction, dispatch.cloud_optical_depth,
            dispatch.cloud_single_scattering_albedo, dispatch.cloud_asymmetry};
        for (const void* input : inputs)
            if (input != nullptr && (input == dispatch.upward_wm2 || input == dispatch.downward_wm2))
                return false;
        const auto valid_nonnegative = [](const float* values, std::size_t count)
        {
            for (std::size_t index = 0; index < count; ++index)
                if (!std::isfinite(values[index]) || values[index] < 0.0f) return false;
            return true;
        };
        if (!valid_nonnegative(dispatch.absorption_optical_depth, spectral_count) ||
            !valid_nonnegative(dispatch.rayleigh_optical_depth, spectral_count) ||
            !valid_nonnegative(dispatch.solar_source_fraction, g_points) ||
            (all_cloud && !valid_nonnegative(dispatch.cloud_optical_depth, cloud_count)))
            return false;
        float source_sum = 0.0f;
        for (std::size_t g = 0; g < g_points; ++g) source_sum += dispatch.solar_source_fraction[g];
        if (!std::isfinite(source_sum) || source_sum <= 0.0f) return false;
        if (all_cloud)
            for (std::size_t index = 0; index < cloud_count; ++index)
                if (!std::isfinite(dispatch.cloud_single_scattering_albedo[index]) ||
                    dispatch.cloud_single_scattering_albedo[index] < 0.0f ||
                    dispatch.cloud_single_scattering_albedo[index] > 1.0f ||
                    !std::isfinite(dispatch.cloud_asymmetry[index]) ||
                    dispatch.cloud_asymmetry[index] < -1.0f || dispatch.cloud_asymmetry[index] > 1.0f)
                    return false;

        try
        {
            std::vector<float> upward(layers + 1, 0.0f);
            std::vector<float> downward(layers + 1, 0.0f);
            if (dispatch.cosine_zenith == 0.0f || dispatch.solar_irradiance_wm2 == 0.0f)
            {
                const std::size_t bytes = (layers + 1) * sizeof(float);
                std::memcpy(dispatch.upward_wm2, upward.data(), bytes);
                std::memcpy(dispatch.downward_wm2, downward.data(), bytes);
                return true;
            }

            std::vector<float> zero_clouds;
            const float* cloud_inputs[3] = {
                dispatch.cloud_optical_depth, dispatch.cloud_single_scattering_albedo,
                dispatch.cloud_asymmetry};
            if (!all_cloud)
            {
                zero_clouds.assign(cloud_count, 0.0f);
                for (const float*& input : cloud_inputs) input = zero_clouds.data();
            }
            const std::size_t input_lengths[6] = {
                spectral_count * sizeof(float), spectral_count * sizeof(float),
                g_points * sizeof(float), cloud_count * sizeof(float),
                cloud_count * sizeof(float), cloud_count * sizeof(float)};
            const void* concrete_inputs[6] = {
                dispatch.absorption_optical_depth, dispatch.rayleigh_optical_depth,
                dispatch.solar_source_fraction, cloud_inputs[0], cloud_inputs[1], cloud_inputs[2]};
            id<MTLBuffer> input_buffers[6];
            for (NSUInteger index = 0; index < 6; ++index)
            {
                input_buffers[index] = [impl_->hardware.device newBufferWithBytes:concrete_inputs[index]
                                                                    length:input_lengths[index]
                                                                   options:MTLResourceStorageModeShared];
                if (input_buffers[index] == nil) return false;
            }
            id<MTLBuffer> output_buffers[12];
            for (NSUInteger index = 0; index < 12; ++index)
            {
                output_buffers[index] =
                    [impl_->hardware.device newBufferWithLength:partial_bytes
                                                        options:MTLResourceStorageModeShared];
                if (output_buffers[index] == nil) return false;
            }

            id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
            if (command_buffer == nil) return false;
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) return false;
            RrtmgShortwaveTransferConstants constants{};
            constants.layers = dispatch.layers;
            constants.has_clouds = all_cloud ? 1 : 0;
            constants.cosine_zenith = dispatch.cosine_zenith;
            constants.solar_irradiance_wm2 = dispatch.solar_irradiance_wm2;
            constants.surface_albedo = dispatch.surface_albedo;
            constants.source_sum = source_sum;
            [encoder setComputePipelineState:impl_->pipeline(KernelId::RrtmgShortwaveTransfer)];
            for (NSUInteger index = 0; index < 6; ++index)
                [encoder setBuffer:input_buffers[index] offset:0 atIndex:index];
            for (NSUInteger index = 0; index < 12; ++index)
                [encoder setBuffer:output_buffers[index] offset:0 atIndex:6 + index];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:18];
            const NSUInteger maximum =
                impl_->pipeline(KernelId::RrtmgShortwaveTransfer).maxTotalThreadsPerThreadgroup;
            if (maximum == 0)
            {
                [encoder endEncoding];
                return false;
            }
            const NSUInteger width = std::min<NSUInteger>(64, maximum);
            [encoder dispatchThreads:MTLSizeMake(g_points, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            if (!impl_->finish(command_buffer)) return false;

            const float* partial_up = static_cast<const float*>(output_buffers[0].contents);
            const float* partial_down = static_cast<const float*>(output_buffers[1].contents);
            for (std::size_t g = 0; g < g_points; ++g)
                for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                {
                    const std::size_t index = g * (layers + 1) + interface_index;
                    upward[interface_index] += partial_up[index];
                    downward[interface_index] += partial_down[index];
                }
            for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                if (!std::isfinite(upward[interface_index]) || upward[interface_index] < 0.0f ||
                    !std::isfinite(downward[interface_index]) || downward[interface_index] < 0.0f)
                    return false;
            const std::size_t result_bytes = (layers + 1) * sizeof(float);
            std::memcpy(dispatch.upward_wm2, upward.data(), result_bytes);
            std::memcpy(dispatch.downward_wm2, downward.data(), result_bytes);
            ++impl_->telemetry.completed_rrtmg_shortwave_transfer_dispatches;
            return true;
        }
        catch (const std::bad_alloc&) { return false; }
        catch (const std::length_error&) { return false; }
    }
}

bool MetalComputeBackend::dispatch_rrtmg_longwave_transfer(
    const RrtmgLongwaveTransferDispatch& dispatch)
{
    @autoreleasepool
    {
        constexpr std::size_t g_points = 140;
        constexpr std::size_t bands = 16;
        if (!supports(RadiationKernelOp::RrtmgLongwaveTransfer) ||
            dispatch.layers <= 0 || dispatch.absorption_optical_depth == nullptr ||
            dispatch.planck_fraction == nullptr || dispatch.layer_planck == nullptr ||
            dispatch.interface_planck == nullptr || dispatch.surface_planck == nullptr ||
            dispatch.band_width_cm == nullptr || dispatch.upward_wm2 == nullptr ||
            dispatch.downward_wm2 == nullptr || dispatch.upward_wm2 == dispatch.downward_wm2 ||
            dispatch.layers == std::numeric_limits<int>::max() ||
            !std::isfinite(dispatch.surface_emissivity) || dispatch.surface_emissivity < 0.0f ||
            dispatch.surface_emissivity > 1.0f || !std::isfinite(dispatch.precipitable_water_cm) ||
            dispatch.precipitable_water_cm < 0.0f)
        {
            return false;
        }
        const std::size_t layers = static_cast<std::size_t>(dispatch.layers);
        if (layers > (std::numeric_limits<std::size_t>::max() / g_points) - 1 ||
            layers > (std::numeric_limits<std::size_t>::max() / bands) - 1)
            return false;
        const std::size_t spectral_count = layers * g_points;
        const std::size_t layer_band_count = layers * bands;
        const std::size_t interface_band_count = (layers + 1) * bands;
        const std::size_t partial_count = (layers + 1) * g_points;
        if (partial_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            interface_band_count > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return false;
        const std::size_t partial_bytes = partial_count * sizeof(float);
        if (partial_bytes > impl_->max_buffer_bytes()) return false;
        const void* inputs[7] = {
            dispatch.absorption_optical_depth, dispatch.planck_fraction, dispatch.layer_planck,
            dispatch.interface_planck, dispatch.surface_planck, dispatch.band_width_cm,
            dispatch.cloud_absorption_optical_depth};
        for (const void* input : inputs)
            if (input != nullptr && (input == dispatch.upward_wm2 || input == dispatch.downward_wm2))
                return false;
        const auto valid_nonnegative = [](const float* values, std::size_t count)
        {
            for (std::size_t index = 0; index < count; ++index)
                if (!std::isfinite(values[index]) || values[index] < 0.0f) return false;
            return true;
        };
        if (!valid_nonnegative(dispatch.absorption_optical_depth, spectral_count) ||
            !valid_nonnegative(dispatch.planck_fraction, spectral_count) ||
            !valid_nonnegative(dispatch.layer_planck, layer_band_count) ||
            !valid_nonnegative(dispatch.interface_planck, interface_band_count) ||
            !valid_nonnegative(dispatch.surface_planck, bands) ||
            !valid_nonnegative(dispatch.band_width_cm, bands) ||
            (dispatch.cloud_absorption_optical_depth != nullptr &&
             !valid_nonnegative(dispatch.cloud_absorption_optical_depth, layer_band_count)))
            return false;
        for (std::size_t index = 0; index < bands; ++index)
            if (dispatch.band_width_cm[index] <= 0.0f) return false;

        try
        {
            std::vector<float> zero_clouds;
            const float* cloud = dispatch.cloud_absorption_optical_depth;
            if (cloud == nullptr)
            {
                zero_clouds.assign(layer_band_count, 0.0f);
                cloud = zero_clouds.data();
            }
            const std::size_t input_lengths[7] = {
                spectral_count * sizeof(float), spectral_count * sizeof(float),
                layer_band_count * sizeof(float), interface_band_count * sizeof(float),
                bands * sizeof(float), bands * sizeof(float), layer_band_count * sizeof(float)};
            const void* concrete_inputs[7] = {
                dispatch.absorption_optical_depth, dispatch.planck_fraction, dispatch.layer_planck,
                dispatch.interface_planck, dispatch.surface_planck, dispatch.band_width_cm, cloud};
            id<MTLBuffer> input_buffers[7];
            for (NSUInteger index = 0; index < 7; ++index)
            {
                input_buffers[index] = [impl_->hardware.device newBufferWithBytes:concrete_inputs[index]
                                                                    length:input_lengths[index]
                                                                   options:MTLResourceStorageModeShared];
                if (input_buffers[index] == nil) return false;
            }
            id<MTLBuffer> upward_partial =
                [impl_->hardware.device newBufferWithLength:partial_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> downward_partial =
                [impl_->hardware.device newBufferWithLength:partial_bytes options:MTLResourceStorageModeShared];
            if (upward_partial == nil || downward_partial == nil) return false;

            id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
            if (command_buffer == nil) return false;
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) return false;
            RrtmgLongwaveTransferConstants constants{};
            constants.layers = dispatch.layers;
            constants.has_clouds = dispatch.cloud_absorption_optical_depth != nullptr ? 1 : 0;
            constants.surface_emissivity = dispatch.surface_emissivity;
            constants.precipitable_water_cm = dispatch.precipitable_water_cm;
            [encoder setComputePipelineState:impl_->pipeline(KernelId::RrtmgLongwaveTransfer)];
            for (NSUInteger index = 0; index < 7; ++index)
                [encoder setBuffer:input_buffers[index] offset:0 atIndex:index];
            [encoder setBuffer:upward_partial offset:0 atIndex:7];
            [encoder setBuffer:downward_partial offset:0 atIndex:8];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:9];
            const NSUInteger maximum =
                impl_->pipeline(KernelId::RrtmgLongwaveTransfer).maxTotalThreadsPerThreadgroup;
            if (maximum == 0)
            {
                [encoder endEncoding];
                return false;
            }
            const NSUInteger width = std::min<NSUInteger>(64, maximum);
            [encoder dispatchThreads:MTLSizeMake(g_points, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            if (!impl_->finish(command_buffer)) return false;

            const float* partial_up = static_cast<const float*>(upward_partial.contents);
            const float* partial_down = static_cast<const float*>(downward_partial.contents);
            std::vector<float> upward(layers + 1, 0.0f);
            std::vector<float> downward(layers + 1, 0.0f);
            for (std::size_t g = 0; g < g_points; ++g)
                for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                {
                    const std::size_t index = g * (layers + 1) + interface_index;
                    upward[interface_index] += partial_up[index];
                    downward[interface_index] += partial_down[index];
                }
            for (std::size_t interface_index = 0; interface_index <= layers; ++interface_index)
                if (!std::isfinite(upward[interface_index]) || upward[interface_index] < 0.0f ||
                    !std::isfinite(downward[interface_index]) || downward[interface_index] < 0.0f)
                    return false;
            const std::size_t result_bytes = (layers + 1) * sizeof(float);
            std::memcpy(dispatch.upward_wm2, upward.data(), result_bytes);
            std::memcpy(dispatch.downward_wm2, downward.data(), result_bytes);
            ++impl_->telemetry.completed_rrtmg_longwave_transfer_dispatches;
            return true;
        }
        catch (const std::bad_alloc&) { return false; }
        catch (const std::length_error&) { return false; }
    }
}
