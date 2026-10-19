#include "catch2/catch.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<std::string> raw_print_lines(const std::filesystem::path& root)
{
    if (!std::filesystem::is_directory(root))
        throw std::runtime_error("Logging scan source directory is missing: " + root.string());

    std::vector<std::string> matches;
    std::size_t              files = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file())
            continue;
        const auto& path = entry.path();
        if (path.extension() != ".cpp" && path.extension() != ".hpp")
            continue;
        // These standalone table tools expose stdout/stderr as their CLI protocol.
        if (path == "src/tools/thompson_activation_table_tool.cpp" ||
            path == "src/tools/thompson_collection_table_tool.cpp" || path == "src/tools/thompson_cold_table_tool.cpp")
            continue;

        ++files;
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("Cannot read logging scan source: " + path.string());
        std::string line;
        std::size_t number = 0;
        while (std::getline(input, line))
        {
            ++number;
            if (line.find("std::cout") != std::string::npos || line.find("std::cerr") != std::string::npos)
                matches.push_back(path.generic_string() + ":" + std::to_string(number) + ": " + line);
        }
        if (input.bad())
            throw std::runtime_error("Failed reading logging scan source: " + path.string());
    }
    if (files == 0)
        throw std::runtime_error("Logging scan found no source files: " + root.string());
    return matches;
}

} // namespace

TEST_CASE("Logging ratchet: raw std::cout/cerr count below threshold", "[diagnostics][logging][ratchet]")
{
    // Never raise this ceiling; migrations to tmv::log_* should reduce it.
    constexpr std::size_t threshold = 55;
    const char*           roots[]   = {
        "src/core/runtime",
        "src/numerics",
        "src/soundings",
        "src/microphysics",
        "src/dynamics",
        "src/radiation",
        "src/boundary_layer",
        "src/turbulence",
        "src/chaos",
        "src/terrain",
        "src/radar",
        "src/diagnostics",
        "src/compute",
        "src/tools",
    };
    std::vector<std::string> matches;
    for (const auto* root : roots)
    {
        const auto found = raw_print_lines(root);
        matches.insert(matches.end(), found.begin(), found.end());
    }
    CAPTURE(matches, threshold);
    CHECK(matches.size() <= threshold);
}

TEST_CASE("Logging ratchet rejects an unavailable source root", "[diagnostics][logging][ratchet]")
{
    REQUIRE_THROWS_AS(raw_print_lines({}), std::runtime_error);
}
