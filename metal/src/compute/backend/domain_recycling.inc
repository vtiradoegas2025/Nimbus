// Homogeneous domain-shift dispatch and storage-mode qualification.
bool MetalComputeBackend::supports_homogeneous_domain_field_shift() const
{
    return ready(nimbus::compute::KernelId::DomainRecyclingHomogeneousFieldShift);
}

bool MetalComputeBackend::dispatch_homogeneous_domain_field_shift(const HomogeneousDomainFieldShiftDispatch& dispatch)
{
    @autoreleasepool
    {
        std::size_t cells       = 0;
        std::size_t field_bytes = 0;
        if (!supports_homogeneous_domain_field_shift() || !valid_dispatch(dispatch, cells, field_bytes))
        {
            return false;
        }
        if (static_cast<std::size_t>(dispatch.nz) > std::numeric_limits<std::size_t>::max() / sizeof(float))
        {
            return false;
        }
        const std::size_t profile_bytes = nimbus::compute::domain_shift::leading_count(dispatch, cells) * sizeof(float);

        id<MTLBuffer> source      = [impl_->hardware.device newBufferWithBytes:dispatch.source
                                                               length:field_bytes
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> profile     = [impl_->hardware.device newBufferWithBytes:nimbus::compute::domain_shift::leading_values(dispatch)
                                                               length:profile_bytes
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> destination = [impl_->hardware.device newBufferWithLength:field_bytes
                                                               options:MTLResourceStorageModeShared];
        if (source == nil || profile == nil || destination == nil)
            return false;

        id<MTLCommandBuffer>         command_buffer = impl_->command_buffer();
        id<MTLComputeCommandEncoder> encoder        = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil)
            return false;

        PushConstants constants{};
        constants.nr                     = dispatch.nr;
        constants.nth                    = dispatch.nth;
        constants.nz                     = dispatch.nz;
        constants.shift_i_cells          = dispatch.shift_i_cells;
        constants.shift_j_cells          = dispatch.shift_j_cells;
        constants.destination_i_begin    = dispatch.destination_i_begin;
        constants.destination_i_end      = dispatch.destination_i_end;
        constants.destination_j_begin    = dispatch.destination_j_begin;
        constants.destination_j_end      = dispatch.destination_j_end;
        constants.leading_profile_offset = 0;
        constants.leading_is_field = dispatch.leading_field != nullptr;

        [encoder setComputePipelineState:impl_->pipeline(KernelId::DomainRecyclingHomogeneousFieldShift)];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:profile offset:0 atIndex:1];
        [encoder setBuffer:destination offset:0 atIndex:2];
        [encoder setBytes:&constants length:sizeof(constants) atIndex:3];

        const NSUInteger maximum = impl_->pipeline(KernelId::DomainRecyclingHomogeneousFieldShift).maxTotalThreadsPerThreadgroup;
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

        // The caller sees a commit only after Metal reports successful
        // completion. This also keeps the destination unchanged on GPU errors.
        std::memcpy(dispatch.destination, destination.contents, field_bytes);
        ++impl_->telemetry.completed_shift_dispatches;
        return true;
    }
}

bool MetalComputeBackend::dispatch_homogeneous_domain_field_shift_batch(
    const HomogeneousDomainFieldShiftBatchDispatch& batch)
{
    @autoreleasepool
    {
        constexpr std::size_t kMaximumFields = 64;
        if (!supports_homogeneous_domain_field_shift() || batch.fields == nullptr || batch.resident_ids == nullptr ||
            batch.count < 1 || static_cast<std::size_t>(batch.count) > kMaximumFields)
        {
            return false;
        }

        const auto& first       = batch.fields[0];
        std::size_t cells       = 0;
        std::size_t field_bytes = 0;
        if (!valid_dispatch(first, cells, field_bytes) ||
            field_bytes > impl_->max_buffer_bytes())
        {
            return false;
        }

        for (int field = 0; field < batch.count; ++field)
        {
            const auto& dispatch    = batch.fields[field];
            const char* resident_id = batch.resident_ids[field];
            std::size_t field_cells = 0;
            std::size_t bytes       = 0;
            if (!valid_dispatch(dispatch, field_cells, bytes) || resident_id == nullptr || resident_id[0] == '\0' ||
                dispatch.nr != first.nr || dispatch.nth != first.nth || dispatch.nz != first.nz ||
                dispatch.shift_i_cells != first.shift_i_cells || dispatch.shift_j_cells != first.shift_j_cells ||
                field_cells != cells || bytes != field_bytes)
            {
                return false;
            }
            for (int prior = 0; prior < field; ++prior)
            {
                const auto& previous = batch.fields[prior];
                if (std::strcmp(resident_id, batch.resident_ids[prior]) == 0 || dispatch.source == previous.source ||
                    dispatch.destination == previous.destination || dispatch.source == previous.destination ||
                    dispatch.destination == previous.source)
                {
                    return false;
                }
            }
        }

        std::array<std::int32_t, kMaximumFields> profile_offsets{};
        std::uint64_t profile_values = 0;
        for (int field = 0; field < batch.count; ++field)
        {
            profile_offsets[field] = static_cast<std::int32_t>(profile_values);
            profile_values += nimbus::compute::domain_shift::leading_count(batch.fields[field], cells);
            if (profile_values > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
                return false;
        }
        if (profile_values == 0 ||
            profile_values > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            profile_values > std::numeric_limits<std::size_t>::max() / sizeof(float))
        {
            return false;
        }
        const std::size_t profile_bytes = static_cast<std::size_t>(profile_values) * sizeof(float);
        if (profile_bytes > impl_->max_buffer_bytes())
            return false;

        id<MTLBuffer> profiles = [impl_->hardware.device newBufferWithLength:profile_bytes options:MTLResourceStorageModeShared];
        if (profiles == nil || profiles.contents == nullptr)
            return false;
        auto* packed_profiles = static_cast<float*>(profiles.contents);
        for (int field = 0; field < batch.count; ++field)
        {
            std::memcpy(packed_profiles + profile_offsets[field],
                        nimbus::compute::domain_shift::leading_values(batch.fields[field]),
                        nimbus::compute::domain_shift::leading_count(batch.fields[field], cells) * sizeof(float));
        }

        std::array<std::size_t, kMaximumFields>              resident_sizes{};
        std::array<Impl::ResidentAllocation, kMaximumFields> resident_allocations{};
        resident_allocations.fill(Impl::ResidentStore::invalid_allocation);
        for (int field = 0; field < batch.count; ++field)
            resident_sizes[static_cast<std::size_t>(field)] = field_bytes;
        if (!impl_->resident.fields.acquire_many(batch.resident_ids,
                                                 resident_sizes.data(),
                                                 static_cast<std::size_t>(batch.count),
                                                 resident_allocations.data()))
        {
            return false;
        }

        const nimbus::compute::FieldExtent extent{cells, sizeof(float)};
        std::array<bool, kMaximumFields>   reusable{};
        std::uint64_t                      uploads = 0;
        std::uint64_t                      reuses  = 0;
        for (int field = 0; field < batch.count; ++field)
        {
            const std::size_t index = static_cast<std::size_t>(field);
            auto*             slot  = impl_->resident_slot(resident_allocations[index]);
            if (slot == nullptr || slot->current == nil || slot->candidate == nil || slot->bytes < field_bytes ||
                slot->current.contents == nullptr || slot->candidate.contents == nullptr)
            {
                return false;
            }
            reusable[index] =
                impl_->resident.fields.reusable(batch.resident_ids[field], batch.fields[field].source, extent, true);
            if (reusable[index])
            {
                ++reuses;
            }
            else
            {
                impl_->resident.fields.invalidate(batch.resident_ids[field]);
                std::memcpy(slot->current.contents, batch.fields[field].source, field_bytes);
                ++uploads;
            }
        }

        id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
        if (command_buffer == nil)
            return false;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil)
            return false;

        const NSUInteger maximum = impl_->pipeline(KernelId::DomainRecyclingHomogeneousFieldShift).maxTotalThreadsPerThreadgroup;
        if (maximum == 0)
        {
            [encoder endEncoding];
            return false;
        }
        const NSUInteger width = std::min<NSUInteger>(64, maximum);
        [encoder setComputePipelineState:impl_->pipeline(KernelId::DomainRecyclingHomogeneousFieldShift)];
        [encoder setBuffer:profiles offset:0 atIndex:1];
        for (int field = 0; field < batch.count; ++field)
        {
            const std::size_t index    = static_cast<std::size_t>(field);
            auto*             slot     = impl_->resident_slot(resident_allocations[index]);
            const auto&       dispatch = batch.fields[field];
            PushConstants     constants{};
            constants.nr                     = dispatch.nr;
            constants.nth                    = dispatch.nth;
            constants.nz                     = dispatch.nz;
            constants.shift_i_cells          = dispatch.shift_i_cells;
            constants.shift_j_cells          = dispatch.shift_j_cells;
            constants.destination_i_begin    = dispatch.destination_i_begin;
            constants.destination_i_end      = dispatch.destination_i_end;
            constants.destination_j_begin    = dispatch.destination_j_begin;
            constants.destination_j_end      = dispatch.destination_j_end;
            constants.leading_profile_offset = profile_offsets[field];
            constants.leading_is_field = dispatch.leading_field != nullptr;

            [encoder setBuffer:slot->current offset:0 atIndex:0];
            [encoder setBuffer:slot->candidate offset:0 atIndex:2];
            [encoder setBytes:&constants length:sizeof(constants) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(cells, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        }
        [encoder endEncoding];

        if (!impl_->finish(command_buffer))
            return false;

        for (int field = 0; field < batch.count; ++field)
        {
            auto* slot = impl_->resident_slot(resident_allocations[static_cast<std::size_t>(field)]);
            std::swap(slot->current, slot->candidate);
        }
        for (int field = 0; field < batch.count; ++field)
        {
            const std::size_t index = static_cast<std::size_t>(field);
            auto*             slot  = impl_->resident_slot(resident_allocations[index]);
            std::memcpy(batch.fields[field].destination, slot->current.contents, field_bytes);
            impl_->resident.fields.note_device_write(
                batch.resident_ids[field], batch.fields[field].destination, extent);
            impl_->resident.fields.note_host_synced(batch.resident_ids[field]);
        }
        impl_->telemetry.completed_shift_dispatches += static_cast<std::uint64_t>(batch.count);
        ++impl_->telemetry.completed_shift_batches;
        impl_->telemetry.resident_uploads += uploads;
        impl_->telemetry.resident_reuses += reuses;
        return true;
    }
}
