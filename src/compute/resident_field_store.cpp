/**
 * @file resident_field_store.cpp
 * @brief Resident allocation ownership and authority-state implementation.
 */

#include "compute/resident_field_store.hpp"

#include <cstring>
#include <limits>
#include <utility>

namespace nimbus::compute
{

ResidentFieldStore::ResidentFieldStore(ResidentFieldStore&& other) noexcept
    : entries_(std::move(other.entries_)),
      residency_(std::move(other.residency_)),
      acquire_(std::exchange(other.acquire_, nullptr)),
      acquire_many_(std::exchange(other.acquire_many_, nullptr)),
      release_(std::exchange(other.release_, nullptr)),
      allocator_data_(std::exchange(other.allocator_data_, nullptr)),
      missing_indices_scratch_(
          std::move(other.missing_indices_scratch_)),
      missing_sizes_scratch_(std::move(other.missing_sizes_scratch_)),
      replaced_allocations_scratch_(
          std::move(other.replaced_allocations_scratch_)),
      candidate_allocations_scratch_(
          std::move(other.candidate_allocations_scratch_)),
      inserted_indices_scratch_(
          std::move(other.inserted_indices_scratch_))
{
    // Standard containers are only required to be valid after a move. Clear
    // explicitly so the source cannot retain duplicate claims.
    other.entries_.clear();
    other.residency_.clear();
}

bool ResidentFieldStore::set_allocator(AcquireFn acquire, ReleaseFn release,
                                       void* user_data)
{
    if (!entries_.empty() &&
        (acquire_ != acquire || release_ != release ||
         allocator_data_ != user_data))
    {
        return false;
    }
    acquire_ = acquire;
    release_ = release;
    allocator_data_ = user_data;
    return true;
}

bool ResidentFieldStore::set_batch_allocator(AcquireManyFn acquire_many)
{
    if (!entries_.empty() && acquire_many_ != acquire_many) return false;
    acquire_many_ = acquire_many;
    return true;
}

void ResidentFieldStore::begin_step() noexcept
{
    residency_.begin_step();
}

std::uint64_t ResidentFieldStore::step_generation() const noexcept
{
    return residency_.step_generation();
}

ResidentFieldStore::Allocation ResidentFieldStore::acquire(
    std::string_view id, std::size_t bytes)
{
    if (id.empty() || bytes == 0 || acquire_ == nullptr)
        return invalid_allocation;
    auto it = entries_.find(id);
    if (it != entries_.end() &&
        it->second.allocation != invalid_allocation &&
        it->second.capacity_bytes >= bytes)
    {
        return it->second.allocation;
    }

    // Materialize a map node before asking the external allocator for a
    // handle. This prevents a later key-allocation failure from leaking a
    // successfully acquired backend allocation.
    const bool inserted = it == entries_.end();
    if (inserted) it = entries_.try_emplace(std::string(id)).first;

    Allocation candidate = invalid_allocation;
    try
    {
        candidate = acquire_(allocator_data_, bytes);
    }
    catch (...)
    {
        if (inserted) entries_.erase(it);
        throw;
    }
    if (candidate == invalid_allocation)
    {
        if (inserted) entries_.erase(it);
        return invalid_allocation;
    }

    Entry& entry = it->second;
    const Allocation replaced = entry.allocation;
    entry = {};
    entry.allocation = candidate;
    entry.capacity_bytes = bytes;
    residency_.invalidate_device(id);
    if (replaced != invalid_allocation && release_ != nullptr)
        release_(allocator_data_, replaced);
    return candidate;
}

bool ResidentFieldStore::acquire_many(
    const char* const* ids, const std::size_t* bytes, std::size_t count,
    Allocation* allocations)
{
    if (ids == nullptr || bytes == nullptr || allocations == nullptr ||
        count == 0 || acquire_ == nullptr)
    {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i)
        allocations[i] = invalid_allocation;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (ids[i] == nullptr || ids[i][0] == '\0' || bytes[i] == 0)
            return false;
        for (std::size_t j = 0; j < i; ++j)
            if (std::strcmp(ids[i], ids[j]) == 0) return false;
    }

    missing_indices_scratch_.clear();
    missing_sizes_scratch_.clear();
    replaced_allocations_scratch_.clear();
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto it = entries_.find(ids[i]);
        if (it == entries_.end() ||
            it->second.allocation == invalid_allocation ||
            it->second.capacity_bytes < bytes[i])
        {
            missing_indices_scratch_.push_back(i);
            missing_sizes_scratch_.push_back(bytes[i]);
            replaced_allocations_scratch_.push_back(
                it == entries_.end() ? invalid_allocation
                                     : it->second.allocation);
        }
        else
        {
            allocations[i] = it->second.allocation;
        }
    }
    if (missing_indices_scratch_.empty()) return true;

    candidate_allocations_scratch_.assign(
        missing_indices_scratch_.size(), invalid_allocation);
    inserted_indices_scratch_.clear();
    inserted_indices_scratch_.reserve(missing_indices_scratch_.size());

    // Stage all keys before invoking an external allocator. Once native
    // handles exist, commit performs only allocation-free map lookups and
    // noexcept scalar assignments.
    try
    {
        for (const std::size_t request : missing_indices_scratch_)
        {
            if (entries_.find(ids[request]) != entries_.end()) continue;
            const auto inserted =
                entries_.try_emplace(std::string(ids[request])).second;
            if (inserted) inserted_indices_scratch_.push_back(request);
        }
    }
    catch (...)
    {
        rollback_staged_entries(ids);
        for (std::size_t i = 0; i < count; ++i)
            allocations[i] = invalid_allocation;
        throw;
    }

    bool acquired = false;
    try
    {
        if (acquire_many_ != nullptr)
        {
            acquired = acquire_many_(
                allocator_data_, missing_sizes_scratch_.data(),
                missing_sizes_scratch_.size(),
                candidate_allocations_scratch_.data());
        }
        else
        {
            acquired = true;
            for (std::size_t i = 0;
                 i < candidate_allocations_scratch_.size(); ++i)
            {
                candidate_allocations_scratch_[i] =
                    acquire_(allocator_data_, missing_sizes_scratch_[i]);
                if (candidate_allocations_scratch_[i] == invalid_allocation)
                {
                    acquired = false;
                    break;
                }
            }
        }
    }
    catch (...)
    {
        release_candidates();
        rollback_staged_entries(ids);
        for (std::size_t i = 0; i < count; ++i)
            allocations[i] = invalid_allocation;
        throw;
    }

    if (acquired)
    {
        for (const Allocation candidate : candidate_allocations_scratch_)
        {
            if (candidate == invalid_allocation)
            {
                acquired = false;
                break;
            }
        }
    }
    if (!acquired)
    {
        release_candidates();
        rollback_staged_entries(ids);
        for (std::size_t i = 0; i < count; ++i)
            allocations[i] = invalid_allocation;
        return false;
    }

    for (std::size_t i = 0;
         i < candidate_allocations_scratch_.size(); ++i)
    {
        const std::size_t request = missing_indices_scratch_[i];
        // Every missing key was staged above, before external allocation.
        Entry& entry = entries_.find(ids[request])->second;
        const Allocation replaced = replaced_allocations_scratch_[i];
        entry = {};
        entry.allocation = candidate_allocations_scratch_[i];
        entry.capacity_bytes = bytes[request];
        residency_.invalidate_device(ids[request]);
        allocations[request] = entry.allocation;
        if (replaced != invalid_allocation && release_ != nullptr)
            release_(allocator_data_, replaced);
    }
    return true;
}

bool ResidentFieldStore::reusable(std::string_view id,
                                  const void* host_identity,
                                  FieldExtent extent, bool enabled)
{
    if (!valid_lineage(id, host_identity, extent))
    {
        residency_.invalidate_device(id);
        return false;
    }
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    Entry& entry = it->second;
    const auto generation = step_generation();
    if (entry.observed_generation != generation)
    {
        residency_.note_host_write(id, host_identity, extent);
        entry.observed_generation = generation;
    }
    return enabled && generation != 0 &&
           entry.allocation != invalid_allocation &&
           residency_.device_current(id, host_identity, extent);
}

void ResidentFieldStore::commit_host_mirror(std::string_view id,
                                            const void* host_identity,
                                            FieldExtent extent)
{
    if (!valid_lineage(id, host_identity, extent))
    {
        residency_.invalidate_device(id);
        return;
    }
    auto it = entries_.find(id);
    if (it == entries_.end() ||
        it->second.allocation == invalid_allocation)
    {
        return;
    }
    Entry& entry = it->second;
    entry.observed_generation = step_generation();
    if (!residency_.host_current(id, host_identity, extent))
        residency_.note_host_write(id, host_identity, extent);
    residency_.note_device_synced_from_host(id, host_identity, extent);
}

void ResidentFieldStore::note_device_write(std::string_view id,
                                           const void* host_identity,
                                           FieldExtent extent)
{
    if (!valid_lineage(id, host_identity, extent))
    {
        residency_.invalidate_device(id);
        return;
    }
    auto it = entries_.find(id);
    if (it == entries_.end() ||
        it->second.allocation == invalid_allocation)
    {
        return;
    }
    Entry& entry = it->second;
    entry.observed_generation = step_generation();
    residency_.note_device_write(id, host_identity, extent);
}

void ResidentFieldStore::note_host_synced(std::string_view id)
{
    if (entries_.find(id) == entries_.end()) return;
    const FieldVersionState* state = residency_.find(id);
    if (state != nullptr)
        residency_.note_host_synced_from_device(
            id, state->host_identity, state->extent);
}

const ResidentFieldStore::Entry* ResidentFieldStore::find(
    std::string_view id) const
{
    const auto it = entries_.find(id);
    return it == entries_.end() ? nullptr : &it->second;
}

const FieldVersionState* ResidentFieldStore::version_state(
    std::string_view id) const
{
    return residency_.find(id);
}

void ResidentFieldStore::invalidate(std::string_view id)
{
    residency_.invalidate_device(id);
}

void ResidentFieldStore::note_host_write(std::string_view id)
{
    const auto it = entries_.find(id);
    if (it == entries_.end()) return;
    it->second.observed_generation = step_generation();
    const FieldVersionState* state = residency_.find(id);
    if (state != nullptr && state->host_identity != nullptr &&
        state->extent.elements != 0 && state->extent.element_bytes != 0)
    {
        const void* host_identity = state->host_identity;
        const FieldExtent extent = state->extent;
        residency_.note_host_write(id, host_identity, extent);
    }
    else
    {
        residency_.invalidate_device(id);
    }
}

void ResidentFieldStore::release(std::string_view id)
{
    const auto it = entries_.find(id);
    if (it == entries_.end()) return;
    residency_.invalidate_device(id);
    if (it->second.allocation != invalid_allocation && release_ != nullptr)
        release_(allocator_data_, it->second.allocation);
    entries_.erase(it);
}

std::size_t ResidentFieldStore::invalidate_all()
{
    std::size_t invalidated = 0;
    for (const auto& item : entries_)
        if (residency_.device_current(item.first)) ++invalidated;
    residency_.invalidate_all_device();
    return invalidated;
}

void ResidentFieldStore::clear(bool release_allocations)
{
    if (release_allocations && release_ != nullptr)
        for (const auto& item : entries_)
            if (item.second.allocation != invalid_allocation)
                release_(allocator_data_, item.second.allocation);
    entries_.clear();
    residency_.clear();
}

bool ResidentFieldStore::valid_lineage(std::string_view id,
                                       const void* host_identity,
                                       FieldExtent extent) noexcept
{
    return !id.empty() && host_identity != nullptr &&
           extent.elements != 0 && extent.element_bytes != 0 &&
           extent.elements <=
               std::numeric_limits<std::size_t>::max() /
                   extent.element_bytes;
}

void ResidentFieldStore::release_candidates() noexcept
{
    if (release_ == nullptr) return;
    for (const Allocation candidate : candidate_allocations_scratch_)
        if (candidate != invalid_allocation)
            release_(allocator_data_, candidate);
}

void ResidentFieldStore::rollback_staged_entries(
    const char* const* ids) noexcept
{
    for (const std::size_t request : inserted_indices_scratch_)
    {
        const auto it = entries_.find(ids[request]);
        if (it != entries_.end()) entries_.erase(it);
    }
    inserted_indices_scratch_.clear();
}

} // namespace nimbus::compute
