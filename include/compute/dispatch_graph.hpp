#pragma once

/**
 * @file dispatch_graph.hpp
 * @brief Backend-neutral hazard graph for batched compute dispatches.
 */

#include "compute/kernel_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nimbus::compute
{

struct FieldUse
{
    std::string field_id;
    ResourceAccess access = ResourceAccess::Read;
};

struct DispatchNode
{
    KernelId kernel{};
    std::vector<FieldUse> fields;
    std::vector<std::size_t> dependencies;
};

class DispatchGraph
{
public:
    std::size_t add(KernelId kernel, std::vector<FieldUse> fields)
    {
        DispatchNode node{kernel, normalize(std::move(fields)), {}};
        const std::size_t index = nodes_.size();
        for (const auto& use : node.fields)
        {
            HazardState& state = hazards_[use.field_id];
            const bool reads = use.access == ResourceAccess::Read ||
                               use.access == ResourceAccess::ReadWrite;
            const bool writes = use.access == ResourceAccess::Write ||
                                use.access == ResourceAccess::ReadWrite;
            if (reads && state.last_writer != none) add_dependency(node, state.last_writer);
            if (writes)
            {
                if (state.last_writer != none) add_dependency(node, state.last_writer);
                for (const auto reader : state.readers) add_dependency(node, reader);
                state.readers.clear();
                state.last_writer = index;
            }
            else if (reads)
            {
                state.readers.push_back(index);
            }
        }
        nodes_.push_back(std::move(node));
        return index;
    }

    bool requires_barrier(std::size_t before, std::size_t after) const
    {
        if (after >= nodes_.size()) return false;
        for (const auto dependency : nodes_[after].dependencies)
            if (dependency <= before) return true;
        return false;
    }

    bool depends_on(std::size_t node, std::size_t dependency) const noexcept
    {
        if (node >= nodes_.size() || dependency >= node) return false;
        for (const auto existing : nodes_[node].dependencies)
            if (existing == dependency) return true;
        return false;
    }

    /**
     * Verify the structural invariants needed by a backend scheduler.
     *
     * DispatchGraph deliberately carries hazards, not executable resource
     * bindings. Backends consume this validation when encoding an already
     * bound typed dispatch; a valid graph alone is not executable work.
     */
    bool valid() const noexcept
    {
        for (std::size_t index = 0; index < nodes_.size(); ++index)
        {
            const DispatchNode& node = nodes_[index];
            if (static_cast<std::size_t>(node.kernel) >= kernel_count()) return false;
            for (std::size_t field = 0; field < node.fields.size(); ++field)
            {
                const FieldUse& use = node.fields[field];
                if (use.field_id.empty() ||
                    (use.access != ResourceAccess::Read &&
                     use.access != ResourceAccess::Write &&
                     use.access != ResourceAccess::ReadWrite))
                    return false;
                for (std::size_t prior = 0; prior < field; ++prior)
                    if (node.fields[prior].field_id == use.field_id) return false;
            }
            for (std::size_t dependency = 0; dependency < node.dependencies.size(); ++dependency)
            {
                if (node.dependencies[dependency] >= index) return false;
                for (std::size_t prior = 0; prior < dependency; ++prior)
                    if (node.dependencies[prior] == node.dependencies[dependency]) return false;
            }
        }
        return true;
    }

    const std::vector<DispatchNode>& nodes() const noexcept { return nodes_; }
    void clear() { nodes_.clear(); hazards_.clear(); }

private:
    static constexpr std::size_t none = static_cast<std::size_t>(-1);
    struct HazardState
    {
        std::size_t last_writer = none;
        std::vector<std::size_t> readers;
    };

    static void add_dependency(DispatchNode& node, std::size_t dependency)
    {
        for (const auto existing : node.dependencies)
            if (existing == dependency) return;
        node.dependencies.push_back(dependency);
    }

    static std::vector<FieldUse> normalize(std::vector<FieldUse> fields)
    {
        std::vector<FieldUse> normalized;
        normalized.reserve(fields.size());
        for (auto& use : fields)
        {
            bool merged = false;
            for (auto& existing : normalized)
            {
                if (existing.field_id != use.field_id) continue;
                const auto combined = static_cast<std::uint8_t>(existing.access) |
                                      static_cast<std::uint8_t>(use.access);
                existing.access = static_cast<ResourceAccess>(combined);
                merged = true;
                break;
            }
            if (!merged) normalized.push_back(std::move(use));
        }
        return normalized;
    }

    std::vector<DispatchNode> nodes_;
    std::unordered_map<std::string, HazardState> hazards_;
};

} // namespace nimbus::compute
