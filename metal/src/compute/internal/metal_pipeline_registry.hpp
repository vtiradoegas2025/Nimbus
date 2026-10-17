#pragma once

// Private Objective-C++ pipeline ownership. This header is included only by
// metal_compute_backend.mm after the Metal framework has been imported.

#include "metal_source_catalog.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace nimbus::metal::detail
{

class MetalPipelineRegistry
{
public:
    bool compile_library(id<MTLDevice>          device,
                         MTLCompileOptions*     options,
                         const char*            label,
                         const char*            source,
                         const char*            source_hash,
                         const MetalKernelSpec* kernels,
                         std::size_t            kernel_count,
                         std::string&           error)
    {
        if (device == nil || source == nullptr || source_hash == nullptr || label == nullptr || kernels == nullptr ||
            kernel_count == 0)
        {
            error = "invalid Metal pipeline-library specification";
            return false;
        }
        ensure_storage();

        NSError*       library_error = nil;
        NSString*      msl_source    = [NSString stringWithUTF8String:source];
        id<MTLLibrary> library       = [device newLibraryWithSource:msl_source options:options error:&library_error];
        if (library == nil)
        {
            error = std::string("runtime ") + label +
                    " MSL compilation failed: " + string_from_ns(library_error.localizedDescription);
            return false;
        }

        for (std::size_t kernel_index_in_library = 0; kernel_index_in_library < kernel_count; ++kernel_index_in_library)
        {
            const MetalKernelSpec& kernel        = kernels[kernel_index_in_library];
            NSString*              function_name = [NSString stringWithUTF8String:kernel.entrypoint];
            id<MTLFunction>        function      = [library newFunctionWithName:function_name];
            if (function == nil)
            {
                error = std::string("runtime ") + label + " MSL library is missing kernel " + kernel.entrypoint;
                return false;
            }

            NSError*                    pipeline_error = nil;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&pipeline_error];
            if (pipeline == nil)
            {
                error = std::string("Metal ") + label + " pipeline creation failed for " + kernel.entrypoint + ": " +
                        string_from_ns(pipeline_error.localizedDescription);
                return false;
            }

            const std::size_t index                                = kernel_index(kernel.id);
            pipelines_[[NSNumber numberWithUnsignedInteger:index]] = pipeline;
            source_hashes_[index]                                  = source_hash;
        }

        [libraries_ addObject:library];
        return true;
    }

    bool alias(nimbus::compute::KernelId alias_id, nimbus::compute::KernelId target_id) noexcept
    {
        const auto pipeline = this->pipeline(target_id);
        if (pipeline == nil)
            return false;
        ensure_storage();
        const std::size_t alias_index                                = kernel_index(alias_id);
        const std::size_t target_index                               = kernel_index(target_id);
        pipelines_[[NSNumber numberWithUnsignedInteger:alias_index]] = pipeline;
        source_hashes_[alias_index]                                  = source_hashes_[target_index];
        return true;
    }

    id<MTLComputePipelineState> pipeline(nimbus::compute::KernelId id) const noexcept
    {
        if (pipelines_ == nil || id == nimbus::compute::KernelId::Count)
            return nil;
        return pipelines_[[NSNumber numberWithUnsignedInteger:kernel_index(id)]];
    }

    bool ready(nimbus::compute::KernelId id) const noexcept
    {
        return pipeline(id) != nil;
    }

    nimbus::compute::BackendPipeline handle(nimbus::compute::KernelId id) const noexcept
    {
        if (!ready(id))
            return 0;
        // BackendPipeline is an opaque nonzero token. KernelId + 1 makes the
        // mapping deterministic and removes a second manually numbered list.
        return static_cast<nimbus::compute::BackendPipeline>(kernel_index(id) + 1);
    }

    const std::string& source_hash(nimbus::compute::KernelId id) const noexcept
    {
        static const std::string empty;
        return id == nimbus::compute::KernelId::Count ? empty : source_hashes_[kernel_index(id)];
    }

    void clear() noexcept
    {
        [pipelines_ removeAllObjects];
        [libraries_ removeAllObjects];
        pipelines_ = nil;
        libraries_ = nil;
        for (std::string& hash : source_hashes_)
            hash.clear();
    }

private:
    static std::size_t kernel_index(nimbus::compute::KernelId id) noexcept
    {
        return static_cast<std::size_t>(id);
    }

    static std::string string_from_ns(NSString* value)
    {
        if (value == nil)
            return {};
        const char* utf8 = value.UTF8String;
        return utf8 == nullptr ? std::string{} : std::string(utf8);
    }

    void ensure_storage()
    {
        if (pipelines_ == nil)
            pipelines_ = [[NSMutableDictionary alloc] init];
        if (libraries_ == nil)
            libraries_ = [[NSMutableArray alloc] init];
    }

    NSMutableDictionary<NSNumber*, id<MTLComputePipelineState>>* pipelines_ = nil;
    NSMutableArray<id<MTLLibrary>>*                              libraries_ = nil;
    std::array<std::string, nimbus::compute::kernel_count()>     source_hashes_{};
};

} // namespace nimbus::metal::detail
