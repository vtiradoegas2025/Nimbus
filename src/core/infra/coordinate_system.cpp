/**
 * @file coordinate_system.cpp
 * @brief Implementation of CoordinateSystem name and parser helpers.
 *
 * Kept independent of runtime_config.cpp so that test binaries can link the
 * parser without dragging in the rest of the runtime configuration surface.
 * The corresponding `global_coordinate_system` is defined alongside the other
 * runtime globals in src/core/runtime_config.cpp (or in tests/test_harness.cpp
 * for test binaries) — *not* here.
 */

#include "core/coordinate_system.hpp"

#include <cctype>
#include <string>

namespace
{

/**
 * @brief Returns a lowercase copy of the input string.
 *
 * Local helper rather than reaching into util/string_utils to keep this
 * translation unit self-contained for tests.
 */
std::string to_lower_local(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (const char ch : value)
    {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

}  // namespace

const char* coordinate_system_name(CoordinateSystem system)
{
    switch (system)
    {
        case CoordinateSystem::Cylindrical:
            return "cylindrical";
        case CoordinateSystem::Cartesian:
            return "cartesian";
    }
    // Defensive fallback for values produced by reading raw memory or
    // future enum values not handled here. Returning a real label keeps
    // diagnostic output well-defined.
    return "cylindrical";
}

bool parse_coordinate_system(const std::string& value, CoordinateSystem& system_out)
{
    if (value.empty())
    {
        return false;
    }

    const std::string normalized = to_lower_local(value);

    if (normalized == "cylindrical" || normalized == "cyl")
    {
        system_out = CoordinateSystem::Cylindrical;
        return true;
    }
    if (normalized == "cartesian" || normalized == "cart")
    {
        system_out = CoordinateSystem::Cartesian;
        return true;
    }

    return false;
}

const char* stagger_type_name(StaggerType stagger)
{
    switch (stagger)
    {
        case StaggerType::Collocated:
            return "collocated";
        case StaggerType::CGrid:
            return "c_grid";
    }
    return "collocated";
}

bool parse_stagger_type(const std::string& value, StaggerType& stagger_out)
{
    if (value.empty())
    {
        return false;
    }

    const std::string normalized = to_lower_local(value);

    if (normalized == "collocated" || normalized == "a_grid")
    {
        stagger_out = StaggerType::Collocated;
        return true;
    }
    if (normalized == "c_grid" || normalized == "cgrid")
    {
        stagger_out = StaggerType::CGrid;
        return true;
    }

    return false;
}
