// Canonical kernel-manifest coverage and source identity.
const char* MetalComputeBackend::canonical_kernel_manifest_hash() noexcept
{
    return canonical_manifest_hash_value().c_str();
}

const char* MetalComputeBackend::metal_artifact_source_hash(nimbus::compute::KernelId kernel) noexcept
{
    return detail::metal_source_hash_for(kernel);
}

MetalManifestCoverage MetalComputeBackend::manifest_coverage() const
{
    MetalManifestCoverage coverage;
    coverage.canonical_manifest_hash = canonical_kernel_manifest_hash();
    for (const auto& entry : nimbus::compute::kernel_manifest())
    {
        ++coverage.manifest_kernels;
        const bool  pipeline_ready = ready(entry.id);
        const char* artifact_hash  = metal_artifact_source_hash(entry.id);
        const bool  has_artifact   = artifact_hash != nullptr && std::strlen(artifact_hash) == 16;
        switch (entry.availability)
        {
        case nimbus::compute::KernelAvailability::Active:
            ++coverage.active_kernels;
            if (entry.precision == nimbus::compute::KernelPrecision::Float32StorageFloat64Arithmetic)
            {
                ++coverage.hardware_limited_kernels;
                if (pipeline_ready || has_artifact) ++coverage.unexpected_available_kernels;
                break;
            }
            if (pipeline_ready)
                ++coverage.ready_active_kernels;
            if (!has_artifact)
                ++coverage.missing_artifact_hashes;
            break;
        case nimbus::compute::KernelAvailability::Deferred:
            ++coverage.deferred_kernels;
            if (pipeline_ready)
                ++coverage.unexpected_available_kernels;
            break;
        case nimbus::compute::KernelAvailability::Orphan:
            ++coverage.orphan_kernels;
            if (pipeline_ready || has_artifact)
                ++coverage.unexpected_available_kernels;
            break;
        }
    }
    coverage.complete = coverage.manifest_kernels == nimbus::compute::kernel_count() &&
                        coverage.ready_active_kernels + coverage.hardware_limited_kernels == coverage.active_kernels &&
                        coverage.unexpected_available_kernels == 0 && coverage.missing_artifact_hashes == 0;
    return coverage;
}
