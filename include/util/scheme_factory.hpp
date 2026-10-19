#pragma once

#include "util/scheme_descriptor.hpp"
#include "util/string_utils.hpp"

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tmv
{

// Maps normalized scheme IDs and aliases to module-local constructors.
template <typename Base>
class SchemeRegistry
{
public:
    using FactoryFn = std::function<std::unique_ptr<Base>()>;

    struct Entry
    {
        std::string id;
        FactoryFn   factory;
    };

    struct Alias
    {
        std::string alias;
        std::string canonical;
    };

    // Descriptor-backed registries derive aliases from the public catalog.
    SchemeRegistry(std::initializer_list<Entry> entries,
                   const std::vector<SchemeDescriptor>& descriptors)
        : entries_(entries)
    {
        std::vector<std::string> factory_ids;
        for (const auto& entry : entries_)
        {
            if (entry.id.empty() || entry.id != strutil::trim_and_lower(entry.id))
            {
                throw std::logic_error(
                    "Scheme factory ID must be canonical: " + entry.id);
            }
            if (std::find(factory_ids.begin(), factory_ids.end(), entry.id)
                != factory_ids.end())
            {
                throw std::logic_error("Duplicate scheme factory ID: " + entry.id);
            }
            factory_ids.push_back(entry.id);
        }

        std::vector<std::string> claimed_ids;
        for (const auto& descriptor : descriptors)
        {
            if (descriptor.id.empty()
                || descriptor.id != strutil::trim_and_lower(descriptor.id))
            {
                throw std::logic_error(
                    "Scheme descriptor ID must be canonical: " + descriptor.id);
            }
            if (std::find(claimed_ids.begin(), claimed_ids.end(), descriptor.id)
                != claimed_ids.end())
            {
                throw std::logic_error(
                    "Duplicate scheme descriptor ID or alias: " + descriptor.id);
            }
            claimed_ids.push_back(descriptor.id);

            bool has_factory = false;
            for (const auto& entry : entries_)
            {
                if (entry.id == descriptor.id)
                {
                    has_factory = true;
                    break;
                }
            }
            if (descriptor.constructible && !has_factory)
            {
                throw std::logic_error(
                    "Scheme descriptor has no factory: " + descriptor.id);
            }
            if (!descriptor.constructible && has_factory)
            {
                throw std::logic_error(
                    "Deferred scheme descriptor has a factory: " + descriptor.id);
            }
            for (const auto& alias : descriptor.aliases)
            {
                if (alias.empty() || alias != strutil::trim_and_lower(alias))
                {
                    throw std::logic_error(
                        "Scheme alias must be canonical: " + alias);
                }
                if (std::find(claimed_ids.begin(), claimed_ids.end(), alias)
                    != claimed_ids.end())
                {
                    throw std::logic_error(
                        "Duplicate scheme descriptor ID or alias: " + alias);
                }
                claimed_ids.push_back(alias);
                aliases_.push_back({alias, descriptor.id});
            }
        }
        for (const auto& entry : entries_)
        {
            bool has_descriptor = false;
            for (const auto& descriptor : descriptors)
            {
                if (descriptor.id == entry.id && descriptor.constructible)
                {
                    has_descriptor = true;
                    break;
                }
            }
            if (!has_descriptor)
            {
                throw std::logic_error(
                    "Scheme factory has no descriptor: " + entry.id);
            }
        }
    }

    std::unique_ptr<Base> create(const std::string& module_name,
                                 const std::string& raw_name) const
    {
        const std::string normalized = resolve(raw_name);

        for (const auto& e : entries_)
        {
            if (e.id == normalized)
            {
                return e.factory();
            }
        }

        throw std::runtime_error(
            "Unknown " + module_name + " scheme: " + raw_name +
            " (normalized: " + normalized +
            "). Available: " + available_csv());
    }

    std::vector<std::string> available_ids() const
    {
        std::vector<std::string> ids;
        ids.reserve(entries_.size());
        for (const auto& e : entries_)
        {
            ids.push_back(e.id);
        }
        return ids;
    }

    std::string available_csv() const
    {
        std::ostringstream oss;
        for (std::size_t i = 0; i < entries_.size(); ++i)
        {
            if (i > 0) oss << ", ";
            oss << entries_[i].id;
        }
        return oss.str();
    }

    std::string resolve(const std::string& raw_name) const
    {
        const std::string normalized = strutil::trim_and_lower(raw_name);

        for (const auto& a : aliases_)
        {
            if (a.alias == normalized)
            {
                return a.canonical;
            }
        }
        return normalized;
    }

private:
    std::vector<Entry> entries_;
    std::vector<Alias> aliases_;
};

} // namespace tmv
