#include "catch2/catch.hpp"
#include "core/runtime/generated_metal_unserialized_controls.hpp"
#include "init/config/v0_to_v1.hpp"
#include "init/config/v1_loader.hpp"

#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>

namespace
{

using Config = tmv::init::config::ConfigV1;

Config load_profile(const std::string& path)
{
    const auto loaded = tmv::init::config::load_v1_config(path);
    INFO(path);
    REQUIRE(loaded.ok());
    return *loaded.config;
}

} // namespace

TEST_CASE("Generated Metal typed guard rejects every hash-blind control mutation", "[generated-metal][typed-controls]")
{
    const auto baseline = load_profile("tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml");
    REQUIRE(tmv::runtime::matches_generated_metal_unserialized_defaults(baseline));
    const auto                                      canonical = tmv::init::config::serialize_v1_to_yaml(baseline);
    const std::vector<std::function<void(Config&)>> mutations = {
        [](Config& c) { c.damping.lateral_sponge_enabled = false; },
        [](Config& c) { --c.damping.lateral_width; },
        [](Config& c) { ++c.damping.lateral_width; },
        [](Config& c) { c.damping.lateral_alpha = std::nextafter(0.05, 0.0); },
        [](Config& c) { c.damping.lateral_alpha = std::nextafter(0.05, 1.0); },
        [](Config& c) { c.damping.w_only_top_sponge = false; },
        [](Config& c) { --c.damping.w_only_depth_cells; },
        [](Config& c) { ++c.damping.w_only_depth_cells; },
        [](Config& c) { c.damping.w_only_alpha_max = std::nextafter(0.20, 0.0); },
        [](Config& c) { c.damping.w_only_alpha_max = std::nextafter(0.20, 1.0); },
        [](Config& c) { c.extensions.solver.slow_rk2 = false; },
        [](Config& c) { c.extensions.solver.lid_bc_gamma = std::nextafter(-1.0, 0.0); },
        [](Config& c) { c.extensions.solver.lid_bc_gamma = std::nextafter(-1.0, -2.0); },
        [](Config& c) { c.extensions.solver.axis_u_m1_filter = true; }};
    for (std::size_t index = 0; index < mutations.size(); ++index)
    {
        INFO(index);
        auto candidate = baseline;
        mutations[index](candidate);
        REQUIRE(tmv::init::config::serialize_v1_to_yaml(candidate) == canonical);
        REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
        const auto before = candidate;
        REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
        REQUIRE(candidate.damping.lateral_sponge_enabled == before.damping.lateral_sponge_enabled);
        REQUIRE(candidate.damping.lateral_width == before.damping.lateral_width);
        REQUIRE(candidate.damping.lateral_alpha == before.damping.lateral_alpha);
        REQUIRE(candidate.damping.w_only_top_sponge == before.damping.w_only_top_sponge);
        REQUIRE(candidate.damping.w_only_depth_cells == before.damping.w_only_depth_cells);
        REQUIRE(candidate.damping.w_only_alpha_max == before.damping.w_only_alpha_max);
        REQUIRE(candidate.extensions.solver.slow_rk2 == before.extensions.solver.slow_rk2);
        REQUIRE(candidate.extensions.solver.lid_bc_gamma == before.extensions.solver.lid_bc_gamma);
        REQUIRE(candidate.extensions.solver.axis_u_m1_filter == before.extensions.solver.axis_u_m1_filter);
    }
    REQUIRE(tmv::runtime::matches_generated_metal_unserialized_defaults(baseline));
    REQUIRE(tmv::init::config::serialize_v1_to_yaml(baseline) == canonical);
}

TEST_CASE("Generated Metal typed guard rejects nonfinite and invalid numeric controls",
          "[generated-metal][typed-controls]")
{
    const auto baseline = load_profile("tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml");
    for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               0.0,
                               1.0})
    {
        auto candidate                  = baseline;
        candidate.damping.lateral_alpha = value;
        REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
        candidate                          = baseline;
        candidate.damping.w_only_alpha_max = value;
        REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
        candidate                                = baseline;
        candidate.extensions.solver.lid_bc_gamma = value;
        REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
    }
    for (const int value : {-1, 0, std::numeric_limits<int>::max()})
    {
        auto candidate                  = baseline;
        candidate.damping.lateral_width = value;
        REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
        candidate                            = baseline;
        candidate.damping.w_only_depth_cells = value;
        REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
    }
}

TEST_CASE("Generated Metal guard preserves all 35 retained witness defaults", "[generated-metal][typed-controls]")
{
    for (std::size_t index = 0; index < 35; ++index)
    {
        std::ostringstream path;
        path << "tests/fixtures/reference/catalog/generated_metal_unserialized_controls/group_"
             << std::setfill('0') << std::setw(2) << index << ".yaml";
        const auto baseline = load_profile(path.str());
        REQUIRE_FALSE(baseline.damping.enabled);
        REQUIRE(tmv::runtime::matches_generated_metal_unserialized_defaults(baseline));
    }
}

TEST_CASE("Generated Metal guard preserves the four admitted source controls and rejects their witnesses",
          "[generated-metal][typed-controls]")
{
    const std::string directory =
        "tests/fixtures/reference/catalog/generated_metal_unserialized_controls/";
    for (const auto profile : {"canonical_npy", "canonical_netcdf", "canonical_zfp", "east_npy"})
    {
        const auto baseline = load_profile(directory + profile + ".yaml");
        REQUIRE(tmv::runtime::matches_generated_metal_unserialized_defaults(baseline));
        const auto canonical = tmv::init::config::serialize_v1_to_yaml(baseline);
        for (const auto key : {"damping_lateral_sponge_enabled",
                               "damping_lateral_width",
                               "damping_lateral_alpha",
                               "damping_w_only_top_sponge",
                               "damping_w_only_depth_cells",
                               "damping_w_only_alpha_max",
                               "extensions_solver_slow_rk2",
                               "extensions_solver_lid_bc_gamma",
                               "extensions_solver_axis_u_m1_filter"})
        {
            const auto candidate = load_profile(directory + profile + "_" + key + ".yaml");
            REQUIRE(tmv::init::config::serialize_v1_to_yaml(candidate) == canonical);
            REQUIRE_FALSE(tmv::runtime::matches_generated_metal_unserialized_defaults(candidate));
        }
    }
}
