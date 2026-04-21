/**
 * @file test_logging_ratchet.cpp
 * @brief Guardrail test that counts raw std::cout/std::cerr in source files.
 *
 * Prevents regression: new code should use tmv::log_info/warn/error/debug
 * from util/log.hpp instead of raw stream output. The threshold ratchets
 * down as files are migrated -- lower it when you migrate a file.
 *
 * Counts are for canonical source files only (src/core/runtime/ and src/),
 * excluding tests, vulkan backend (has its own logging), and log.hpp itself.
 *
 * Raw prints inside properly gated blocks (e.g., `if (log_normal_enabled())`)
 * are acceptable for formatted multi-line output but still count toward the
 * threshold. The goal is awareness, not zero.
 */
#include "catch2/catch.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

/// Count raw std::cout/std::cerr occurrences in source files via grep.
/// Returns -1 on failure.
int count_raw_prints()
{
    // Search canonical source directories, excluding vulkan, tests, and log.hpp
    const char* cmd =
        "grep -rn 'std::cout\\|std::cerr' "
        "src/core/runtime/ src/advection/ src/numerics/ src/soundings/ "
        "src/microphysics/ src/dynamics/ src/radiation/ src/boundary_layer/ "
        "src/turbulence/ src/chaos/ src/terrain/ src/radar/ src/diagnostics/ "
        "src/compute/ src/tools/ "
        "--include='*.cpp' --include='*.hpp' "
        "| grep -v 'log\\.hpp' "
        "| wc -l";

    FILE* pipe = popen(cmd, "r");
    if (!pipe) return -1;

    char buf[64];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe))
    {
        result += buf;
    }
    int status = pclose(pipe);
    if (status != 0) return -1;

    return std::atoi(result.c_str());
}

} // namespace

TEST_CASE("Logging ratchet: raw std::cout/cerr count below threshold",
          "[diagnostics][logging][ratchet]")
{
    // Threshold: current count after migrating runtime_config.cpp warnings.
    // Ratchet this DOWN as more files are migrated to tmv::log_*.
    // Do NOT raise this number -- that means new raw prints were added.
    constexpr int THRESHOLD = 274;

    int count = count_raw_prints();
    REQUIRE(count >= 0); // grep must succeed

    CAPTURE(count, THRESHOLD);
    CHECK(count <= THRESHOLD);

    // Informational: show how far we are from the target
    if (count <= THRESHOLD)
    {
        WARN("Raw print count: " + std::to_string(count) +
             " / " + std::to_string(THRESHOLD) +
             " (headroom: " + std::to_string(THRESHOLD - count) + ")");
    }
}
