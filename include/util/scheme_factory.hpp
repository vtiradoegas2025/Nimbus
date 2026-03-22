#pragma once

#include "util/string_utils.hpp"

#include <functional>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * @file scheme_factory.hpp
 * @brief Generic factory template for physics/numerics scheme dispatch.
 *
 * Eliminates the per-module boilerplate that was previously duplicated
 * across 12 factory.cpp files. Each factory now declares a static
 * SchemeRegistry and delegates to it.
 *
 * Usage:
 * @code
 *   #include "util/scheme_factory.hpp"
 *
 *   static const tmv::SchemeRegistry<MyBaseType> registry({
 *       {"scheme_a", [] { return std::make_unique<SchemeA>(); }},
 *       {"scheme_b", [] { return std::make_unique<SchemeB>(); }},
 *   }, {
 *       {"alias1", "scheme_a"},
 *       {"alias2", "scheme_b"},
 *   });
 *
 *   // In create_*_scheme():
 *   return registry.create("dynamics", user_input);
 *
 *   // In get_available_*_schemes():
 *   return registry.available_ids();
 * @endcode
 */

namespace tmv
{

/**
 * @brief A lightweight scheme registry that maps normalized names to
 *        factory callables, with optional alias resolution.
 *
 * @tparam Base The polymorphic base type returned by the factory
 *              (e.g., DynamicsScheme, MicrophysicsScheme).
 */
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

    /**
     * @brief Constructs a registry from a list of scheme entries and
     *        optional aliases.
     *
     * @param entries  Canonical scheme id → factory pairs.
     * @param aliases  Alternative names that map to a canonical id.
     */
    SchemeRegistry(std::initializer_list<Entry> entries,
                   std::initializer_list<Alias> aliases = {})
        : entries_(entries), aliases_(aliases)
    {
    }

    /**
     * @brief Creates a scheme instance from a user-supplied name.
     *
     * The name is trimmed and lowercased, then alias-resolved before
     * looking up in the registry. Throws std::runtime_error on miss.
     *
     * @param module_name Human-readable module name for error messages
     *                    (e.g., "dynamics", "microphysics").
     * @param raw_name    The user/config-supplied scheme name.
     * @return Owning pointer to the constructed scheme.
     */
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

    /**
     * @brief Returns the list of canonical scheme ids.
     */
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

    /**
     * @brief Returns a comma-separated string of available scheme ids.
     */
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

    /**
     * @brief Normalizes and alias-resolves a raw scheme name.
     */
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
