// Private Objective-C++ composition root and translation-unit helpers.
// Included only by ../metal_compute_backend.mm.
using nimbus::compute::KernelId;

namespace
{

std::string ns_string(NSString* value)
{
    if (value == nil)
        return {};
    const char* utf8 = value.UTF8String;
    return utf8 == nullptr ? std::string{} : std::string(utf8);
}

const std::string& canonical_manifest_hash_value()
{
    static const std::string hash = []
    {
        std::ostringstream canonical;
        const auto         append_text = [&canonical](const char* value)
        {
            const char* safe = value == nullptr ? "" : value;
            canonical << std::strlen(safe) << ':' << safe << ';';
        };
        for (const auto& entry : nimbus::compute::kernel_manifest())
        {
            canonical << static_cast<unsigned>(entry.id) << '|';
            append_text(entry.operation_id);
            append_text(entry.kernel_name);
            append_text(entry.artifact_name);
            append_text(entry.artifact_hash);
            canonical << static_cast<unsigned>(entry.resources.descriptor_span) << ',' << entry.resources.read_mask
                      << ',' << entry.resources.write_mask << ',' << static_cast<unsigned>(entry.shape) << ','
                      << entry.constants.bytes << ',' << static_cast<unsigned>(entry.constants.alignment) << ','
                      << static_cast<unsigned>(entry.constants.version) << ',' << static_cast<unsigned>(entry.precision)
                      << ',' << static_cast<unsigned>(entry.cpu_fallback) << ',';
            append_text(entry.cpu_reference);
            canonical << static_cast<unsigned>(entry.parity_tier) << ','
                      << static_cast<unsigned>(entry.comparison_domain) << ','
                      << static_cast<unsigned>(entry.coordinate) << ',' << static_cast<unsigned>(entry.availability)
                      << ',' << (entry.startup_required ? 1 : 0) << '\n';
        }
        return detail::MetalLibrarySpec::hash(canonical.str().c_str());
    }();
    return hash;
}

} // namespace

class MetalComputeBackend::Impl
{
public:
    using ResidentStore      = detail::MetalResidentState::Store;
    using ResidentAllocation = detail::MetalResidentState::Allocation;
    using ResidentSlot       = detail::MetalResidentState::Slot;

    detail::MetalDeviceContext      hardware;
    detail::MetalPipelineRegistry   pipelines;
    detail::MetalSharedAllocations  shared_allocations;
    detail::MetalResidentState      resident;
    detail::MetalTelemetry          telemetry;
    tmv::FlatCGridDeviceScalarAttributionWorkspace scalar_attribution_workspace;
    id<MTLBuffer>                                                    thompson_completion_coefficients = nil;
    std::shared_ptr<const nimbus::compute::ThompsonCompletionTables> thompson_completion_tables;
    id<MTLBuffer> kessler_buffers[13] = {};

    id<MTLBuffer> kessler_buffer(std::size_t index, std::size_t bytes,
                                  const float* source = nullptr)
    {
        auto& buffer = kessler_buffers[index];
        if (buffer == nil || buffer.length < bytes)
        {
            id<MTLBuffer> candidate = [hardware.device newBufferWithLength:bytes
                options:MTLResourceStorageModeShared];
            if (candidate == nil)
                return nil;
            buffer = candidate;
        }
        if (source != nullptr)
            std::memcpy(buffer.contents, source, bytes);
        return buffer;
    }

    ResidentSlot* resident_slot(ResidentAllocation allocation) noexcept
    {
        return resident.slot(allocation);
    }

    id<MTLComputePipelineState> pipeline(KernelId kernel) const noexcept
    {
        return pipelines.pipeline(kernel);
    }

    void clear_generation() noexcept
    {
        for (auto& buffer : kessler_buffers)
            buffer = nil;
        resident.clear();
        shared_allocations.clear();
        pipelines.clear();
        hardware.reset();
        telemetry.reset();
    }

    std::size_t max_buffer_bytes() const noexcept
    {
        return hardware.max_buffer_bytes();
    }

    id<MTLCommandBuffer> command_buffer() const noexcept
    {
        return hardware.command_buffer();
    }

    bool finish(id<MTLCommandBuffer> command_buffer) noexcept
    {
        return hardware.finish(command_buffer, telemetry);
    }
};
