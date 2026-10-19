/**
 * @file vulkan_compute_telemetry.hpp
 * @brief Backend-neutral counters for Vulkan compute profiling output.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <ostream>

namespace tmv::vulkan_compute
{

struct MemoryCounters
{
    std::array<std::uint64_t, 6> values{};
};

class MemoryWindowRecorder
{
public:
    bool configure(const char* token, std::uint64_t first_boundary,
                   std::uint64_t end_boundary_exclusive) noexcept
    {
        if (configured_ || token == nullptr || first_boundary == 0 ||
            end_boundary_exclusive <= first_boundary)
        {
            return false;
        }
        for (std::size_t i = 0; i < 64; ++i)
        {
            if (!((token[i] >= '0' && token[i] <= '9') ||
                  (token[i] >= 'a' && token[i] <= 'f')))
            {
                return false;
            }
        }
        if (token[64] != '\0')
        {
            return false;
        }
        std::copy_n(token, 64, token_.begin());
        first_boundary_ = first_boundary;
        end_boundary_exclusive_ = end_boundary_exclusive;
        configured_ = true;
        return true;
    }

    void boundary(std::uint64_t boundary, const MemoryCounters& counters,
                  bool accounting_valid, std::ostream& output, std::uint64_t pid)
    {
        if (!configured_ || finished_)
        {
            return;
        }
        if (boundary <= last_boundary_ || (last_boundary_ != 0 &&
            boundary != last_boundary_ + 1))
        {
            valid_ = false;
        }
        last_boundary_ = boundary;
        if (!active_ && boundary == first_boundary_)
        {
            begin_ = current_ = high_water_ = counters;
            valid_ = valid_ && accounting_valid;
            active_ = true;
            emit(output, pid, "begin", boundary, false);
        }
        else if (active_ && boundary >= end_boundary_exclusive_)
        {
            observe(counters, accounting_valid, false);
            valid_ = valid_ && boundary == end_boundary_exclusive_;
            emit(output, pid, "end", boundary, true);
            active_ = false;
            finished_ = true;
        }
    }

    void observe(const MemoryCounters& counters, bool accounting_valid,
                 bool allocation_event = true) noexcept
    {
        if (!active_)
        {
            return;
        }
        current_ = counters;
        valid_ = valid_ && accounting_valid;
        for (std::size_t i = 0; i < current_.values.size(); ++i)
        {
            high_water_.values[i] = std::max(high_water_.values[i], current_.values[i]);
        }
        if (allocation_event)
        {
            if (events_ == std::numeric_limits<std::uint64_t>::max())
            {
                valid_ = false;
            }
            else
            {
                ++events_;
            }
        }
    }

    void abort(const MemoryCounters& counters, bool accounting_valid,
               std::ostream& output, std::uint64_t pid)
    {
        if (!active_)
        {
            return;
        }
        observe(counters, accounting_valid, false);
        valid_ = false;
        emit(output, pid, "end", last_boundary_, false);
        active_ = false;
        finished_ = true;
    }

private:
    void emit(std::ostream& output, std::uint64_t pid, const char* event,
              std::uint64_t boundary, bool completed)
    {
        struct FormatGuard
        {
            std::ostream& output;
            std::ios_base::fmtflags flags;
            std::streamsize width;
            ~FormatGuard()
            {
                output.flags(flags);
                output.width(width);
            }
        } guard{output, output.flags(), output.width()};
        output.setf(std::ios_base::dec, std::ios_base::basefield);
        output.unsetf(std::ios_base::showbase | std::ios_base::showpos | std::ios_base::uppercase);
        output.width(0);
        constexpr std::array<const char*, 6> names{
            "staging_buffer_extent_bytes", "device_buffer_extent_bytes",
            "staging_memory_reserved_bytes", "device_memory_reserved_bytes",
            "staging_memory_allocation_count", "device_memory_allocation_count"};
        output << "[C6 MEMORY WINDOW] {\"schema\":\"nimbus.c6-vulkan-memory-window-event.v1\","
               << "\"runToken\":\"" << token_.data() << "\",\"pid\":" << pid
               << ",\"owner\":\"VulkanComputeBackend\","
               << "\"source\":\"vulkan/include/compute/vulkan_compute_telemetry.hpp\","
               << "\"scope\":\"backend_residency_boundaries\","
               << "\"aggregation\":\"separate_nonadditive_compartments\","
               << "\"recordCapacity\":2,\"firstBoundary\":" << first_boundary_
               << ",\"endBoundaryExclusive\":" << end_boundary_exclusive_
               << ",\"event\":\"" << event << "\",\"boundary\":" << boundary
               << ",\"completed\":" << (completed ? "true" : "false")
               << ",\"valid\":" << (valid_ ? "true" : "false")
               << ",\"allocationHookEvents\":" << events_ << ",\"metrics\":{";
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            if (i != 0)
            {
                output << ',';
            }
            output << '\"' << names[i] << "\":{\"unit\":\""
                   << (i < 4 ? "bytes" : "allocations") << "\",\"begin\":"
                   << begin_.values[i] << ",\"current\":" << current_.values[i]
                   << ",\"highWater\":" << high_water_.values[i] << '}';
        }
        output << "}}\n";
    }

    std::array<char, 65> token_{};
    MemoryCounters begin_{};
    MemoryCounters current_{};
    MemoryCounters high_water_{};
    std::uint64_t first_boundary_ = 0;
    std::uint64_t end_boundary_exclusive_ = 0;
    std::uint64_t last_boundary_ = 0;
    std::uint64_t events_ = 0;
    bool configured_ = false;
    bool active_ = false;
    bool finished_ = false;
    bool valid_ = true;
};

struct ActivityCounters
{
    std::uint64_t model_steps = 0;
    std::uint64_t submitted_command_buffers = 0;
    std::uint64_t submitted_dispatches = 0;
    std::uint64_t host_upload_bytes = 0;
    std::uint64_t host_download_bytes = 0;
    std::uint64_t device_upload_bytes = 0;
    std::uint64_t device_download_bytes = 0;
    std::uint64_t resident_hits = 0;
    std::uint64_t resident_misses = 0;
    std::uint64_t resident_upload_bytes_saved = 0;
    std::uint64_t writeonly_output_upload_bytes_saved = 0;
    std::uint64_t descriptor_cache_hits = 0;
    std::uint64_t descriptor_cache_misses = 0;
};

inline std::uint64_t counter_delta(std::uint64_t current,
                                   std::uint64_t previous)
{
    return current >= previous ? current - previous : current;
}

inline ActivityCounters activity_delta(const ActivityCounters& current,
                                       const ActivityCounters& previous)
{
    ActivityCounters delta;
    delta.model_steps =
        counter_delta(current.model_steps, previous.model_steps);
    delta.submitted_command_buffers = counter_delta(
        current.submitted_command_buffers, previous.submitted_command_buffers);
    delta.submitted_dispatches = counter_delta(
        current.submitted_dispatches, previous.submitted_dispatches);
    delta.host_upload_bytes =
        counter_delta(current.host_upload_bytes, previous.host_upload_bytes);
    delta.host_download_bytes =
        counter_delta(current.host_download_bytes, previous.host_download_bytes);
    delta.device_upload_bytes =
        counter_delta(current.device_upload_bytes, previous.device_upload_bytes);
    delta.device_download_bytes = counter_delta(
        current.device_download_bytes, previous.device_download_bytes);
    delta.resident_hits =
        counter_delta(current.resident_hits, previous.resident_hits);
    delta.resident_misses =
        counter_delta(current.resident_misses, previous.resident_misses);
    delta.resident_upload_bytes_saved = counter_delta(
        current.resident_upload_bytes_saved,
        previous.resident_upload_bytes_saved);
    delta.writeonly_output_upload_bytes_saved = counter_delta(
        current.writeonly_output_upload_bytes_saved,
        previous.writeonly_output_upload_bytes_saved);
    delta.descriptor_cache_hits = counter_delta(
        current.descriptor_cache_hits, previous.descriptor_cache_hits);
    delta.descriptor_cache_misses = counter_delta(
        current.descriptor_cache_misses, previous.descriptor_cache_misses);
    return delta;
}

inline void attribute_host_readback(ActivityCounters& producer,
                                    ActivityCounters& attributed,
                                    std::uint64_t bytes)
{
    producer.host_download_bytes += bytes;
    attributed.host_download_bytes += bytes;
}

struct Telemetry
{
    std::uint64_t model_steps = 0;
    std::uint64_t submitted_command_buffers = 0;
    std::uint64_t submitted_dispatches = 0;
    std::uint64_t host_upload_bytes = 0;
    std::uint64_t host_download_bytes = 0;
    std::uint64_t device_upload_bytes = 0;
    std::uint64_t device_download_bytes = 0;
    std::uint64_t resident_hits = 0;
    std::uint64_t resident_misses = 0;
    std::uint64_t resident_upload_bytes_saved = 0;
    std::uint64_t writeonly_output_upload_bytes_saved = 0;
    std::uint64_t resident_full_invalidations = 0;
    std::uint64_t resident_entries_invalidated = 0;
    std::uint64_t batched_command_buffers = 0;
    std::uint64_t synchronous_waits = 0;
    std::uint64_t descriptor_cache_hits = 0;
    std::uint64_t descriptor_cache_misses = 0;
    std::uint64_t staging_allocated_bytes = 0;
    std::uint64_t staging_high_water_bytes = 0;
    std::uint64_t device_allocated_bytes = 0;
    std::uint64_t device_high_water_bytes = 0;
    std::uint64_t staging_memory_allocations = 0;
    std::uint64_t staging_memory_allocations_high_water = 0;
    std::uint64_t device_memory_allocations = 0;
    std::uint64_t device_memory_allocations_high_water = 0;
    std::uint64_t staging_memory_reserved_bytes = 0;
    std::uint64_t staging_memory_reserved_high_water_bytes = 0;
    std::uint64_t device_memory_reserved_bytes = 0;
    std::uint64_t device_memory_reserved_high_water_bytes = 0;
    bool memory_accounting_valid = true;
    MemoryWindowRecorder memory_window;

    MemoryCounters memory_counters() const noexcept
    {
        return {{{staging_allocated_bytes, device_allocated_bytes,
                  staging_memory_reserved_bytes, device_memory_reserved_bytes,
                  staging_memory_allocations, device_memory_allocations}}};
    }

    void note_model_step() { ++model_steps; }

    void note_submission(std::uint64_t dispatches)
    {
        ++submitted_command_buffers;
        submitted_dispatches += dispatches;
        if (dispatches > 1) ++batched_command_buffers;
    }

    void note_host_upload(std::uint64_t bytes) { host_upload_bytes += bytes; }
    void note_host_download(std::uint64_t bytes) { host_download_bytes += bytes; }
    void note_device_upload(std::uint64_t bytes) { device_upload_bytes += bytes; }
    void note_device_download(std::uint64_t bytes) { device_download_bytes += bytes; }

    void note_resident_lookup(bool hit, std::uint64_t upload_bytes_saved = 0)
    {
        if (hit)
        {
            ++resident_hits;
            resident_upload_bytes_saved += upload_bytes_saved;
        }
        else
        {
            ++resident_misses;
        }
    }

    void note_writeonly_output_upload_avoided(std::uint64_t bytes)
    {
        writeonly_output_upload_bytes_saved += bytes;
    }

    void note_resident_full_invalidation(std::uint64_t entries)
    {
        ++resident_full_invalidations;
        resident_entries_invalidated += entries;
    }

    void note_synchronous_wait() { ++synchronous_waits; }

    void note_descriptor_lookup(bool hit)
    {
        if (hit) ++descriptor_cache_hits;
        else ++descriptor_cache_misses;
    }

    void note_allocation(std::uint64_t bytes, bool device)
    {
        auto& current = device ? device_allocated_bytes : staging_allocated_bytes;
        auto& high_water = device ? device_high_water_bytes : staging_high_water_bytes;
        memory_accounting_valid = memory_accounting_valid &&
            bytes <= std::numeric_limits<std::uint64_t>::max() - current;
        current += bytes;
        high_water = std::max(high_water, current);
        memory_window.observe(memory_counters(), memory_accounting_valid);
    }

    void note_release(std::uint64_t bytes, bool device)
    {
        auto& current = device ? device_allocated_bytes : staging_allocated_bytes;
        memory_accounting_valid = memory_accounting_valid && bytes <= current;
        current = bytes > current ? 0 : current - bytes;
        memory_window.observe(memory_counters(), memory_accounting_valid);
    }

    void note_memory_allocation(std::uint64_t bytes, bool device)
    {
        auto& count = device ? device_memory_allocations
                             : staging_memory_allocations;
        auto& count_high = device ? device_memory_allocations_high_water
                                  : staging_memory_allocations_high_water;
        auto& reserved = device ? device_memory_reserved_bytes
                                : staging_memory_reserved_bytes;
        auto& reserved_high = device ? device_memory_reserved_high_water_bytes
                                     : staging_memory_reserved_high_water_bytes;
        memory_accounting_valid = memory_accounting_valid &&
            count != std::numeric_limits<std::uint64_t>::max() &&
            bytes <= std::numeric_limits<std::uint64_t>::max() - reserved;
        ++count;
        count_high = std::max(count_high, count);
        reserved += bytes;
        reserved_high = std::max(reserved_high, reserved);
        memory_window.observe(memory_counters(), memory_accounting_valid);
    }

    void note_memory_release(std::uint64_t bytes, bool device)
    {
        auto& count = device ? device_memory_allocations
                             : staging_memory_allocations;
        auto& reserved = device ? device_memory_reserved_bytes
                                : staging_memory_reserved_bytes;
        memory_accounting_valid = memory_accounting_valid && count > 0 && bytes <= reserved;
        if (count > 0) --count;
        reserved = bytes > reserved ? 0 : reserved - bytes;
        memory_window.observe(memory_counters(), memory_accounting_valid);
    }

    ActivityCounters activity() const
    {
        ActivityCounters counters;
        counters.model_steps = model_steps;
        counters.submitted_command_buffers = submitted_command_buffers;
        counters.submitted_dispatches = submitted_dispatches;
        counters.host_upload_bytes = host_upload_bytes;
        counters.host_download_bytes = host_download_bytes;
        counters.device_upload_bytes = device_upload_bytes;
        counters.device_download_bytes = device_download_bytes;
        counters.resident_hits = resident_hits;
        counters.resident_misses = resident_misses;
        counters.resident_upload_bytes_saved = resident_upload_bytes_saved;
        counters.writeonly_output_upload_bytes_saved =
            writeonly_output_upload_bytes_saved;
        counters.descriptor_cache_hits = descriptor_cache_hits;
        counters.descriptor_cache_misses = descriptor_cache_misses;
        return counters;
    }
};

}  // namespace tmv::vulkan_compute
