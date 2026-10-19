#pragma once

/**
 * @file resident_field_store.hpp
 * @brief Backend-neutral resident allocation and authoritative-copy store.
 */

#include "compute/compute_residency.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace nimbus::compute
{

/**
 * Named backend allocations coupled to a field-authority graph.
 *
 * The store owns named claims on allocator handles, but not the allocator or
 * its opaque context. Callbacks must not throw, successful acquisitions must
 * return distinct live handles, and callback bindings must outlive all claims.
 * Configuration changes are accepted only while the store is empty.
 *
 * Destruction does not invoke release callbacks because the external allocator
 * may already be gone. Owners requiring individual release must call
 * clear(true). Instances are not thread-safe; the backend must serialize all
 * allocation and residency transitions.
 *
 * Move construction transfers claims and callback bindings. Move assignment is
 * unavailable because replacing a non-empty store could orphan live claims.
 */
class ResidentFieldStore
{
public:
    using Allocation = std::int64_t;
    static constexpr Allocation invalid_allocation = -1;
    using AcquireFn = Allocation (*)(void*, std::size_t);
    using AcquireManyFn = bool (*)(
        void*, const std::size_t*, std::size_t, Allocation*);
    using ReleaseFn = void (*)(void*, Allocation);

    struct Entry
    {
        Allocation allocation = invalid_allocation;
        std::size_t capacity_bytes = 0;
        std::uint64_t observed_generation = 0;
    };

    ResidentFieldStore() = default;
    ResidentFieldStore(const ResidentFieldStore&) = delete;
    ResidentFieldStore& operator=(const ResidentFieldStore&) = delete;
    ResidentFieldStore(ResidentFieldStore&& other) noexcept;
    ResidentFieldStore& operator=(ResidentFieldStore&&) = delete;

    /**
     * Bind the allocator used by subsequently acquired entries.
     *
     * Repeating the current binding succeeds. Replacing any part while entries
     * are live is rejected. Ignoring the result remains source-compatible with
     * the original configuration-only API.
     */
    bool set_allocator(AcquireFn acquire, ReleaseFn release, void* user_data);

    /** Apply the same live-entry ownership rule to the batch callback. */
    bool set_batch_allocator(AcquireManyFn acquire_many);

    void begin_step() noexcept;
    std::uint64_t step_generation() const noexcept;

    /** Acquire or reuse one named allocation; malformed requests fail closed. */
    Allocation acquire(std::string_view id, std::size_t bytes);

    /**
     * Acquire distinct named entries as one transaction.
     *
     * Existing sufficient entries retain their allocations. Missing or
     * undersized entries are acquired together when a batch callback exists.
     * Failure releases every candidate and publishes no new entry.
     */
    bool acquire_many(const char* const* ids, const std::size_t* bytes,
                      std::size_t count, Allocation* allocations);

    bool reusable(std::string_view id, const void* host_identity,
                  FieldExtent extent, bool enabled);
    void commit_host_mirror(std::string_view id, const void* host_identity,
                            FieldExtent extent);
    void note_device_write(std::string_view id, const void* host_identity,
                           FieldExtent extent);
    void note_host_synced(std::string_view id);

    const Entry* find(std::string_view id) const;
    const FieldVersionState* version_state(std::string_view id) const;

    void invalidate(std::string_view id);
    void note_host_write(std::string_view id);

    /**
     * Relinquish one named allocation after its audited consumer window.
     *
     * The backing pool may immediately reuse the slot. This differs from
     * invalidate(), which preserves exclusive named ownership.
     */
    void release(std::string_view id);

    std::size_t invalidate_all();
    void clear(bool release_allocations = false);

private:
    static bool valid_lineage(std::string_view id, const void* host_identity,
                              FieldExtent extent) noexcept;
    void release_candidates() noexcept;
    void rollback_staged_entries(const char* const* ids) noexcept;

    std::map<std::string, Entry, std::less<>> entries_;
    FieldResidencyGraph residency_;
    AcquireFn acquire_ = nullptr;
    AcquireManyFn acquire_many_ = nullptr;
    ReleaseFn release_ = nullptr;
    void* allocator_data_ = nullptr;
    std::vector<std::size_t> missing_indices_scratch_;
    std::vector<std::size_t> missing_sizes_scratch_;
    std::vector<Allocation> replaced_allocations_scratch_;
    std::vector<Allocation> candidate_allocations_scratch_;
    std::vector<std::size_t> inserted_indices_scratch_;
};

} // namespace nimbus::compute
