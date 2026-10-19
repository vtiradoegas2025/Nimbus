#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace nimbus::output
{

inline constexpr int kQualifiedRadarZfpV4ScopeVersion = 1;
inline constexpr bool kQualifiedRadarZfpV4PublicAdmissionOpen = true;
inline constexpr std::array<std::string_view, 5> kQualifiedRadarZfpV4Products{
    "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
inline constexpr std::array<std::string_view, 8> kQualifiedRadarZfpV4CoreFields{
    "u", "v", "w", "theta", "p", "rho", "qv", "qr"};
inline constexpr std::array<std::string_view, 3> kQualifiedRadarZfpV4FormatTiers{
    "zfp_low", "zfp_medium", "zfp_high"};
inline constexpr std::array<std::string_view, 3> kQualifiedRadarZfpV4Presets{
    "minimal", "radar_ready", "radar_polarimetric_ready"};
inline constexpr std::array<int, 2> kQualifiedRadarZfpV4WriteCadenceSeconds{1, 4};

inline bool is_qualified_radar_zfp_v4_field(std::string_view field)
{
    for (const auto product : kQualifiedRadarZfpV4Products)
    {
        if (field == product)
        {
            return true;
        }
    }
    return false;
}

struct QualifiedRadarZfpV4Context
{
    bool qualified = false;
    bool cpu = false;
    bool fallback_off = false;
    bool cartesian = false;
    bool c_grid = false;
    bool flat = false;
    bool stationary = false;
    bool no_nesting = false;
    bool no_tracking = false;
    bool no_side_stream = false;
    bool no_tiered_cadence = false;
    bool no_live_shm = false;
    bool no_trajectories = false;
    bool deterministic_reductions = false;
    bool reviewed_source_profile = false;
    std::vector<std::string> selected_radar_products;
    std::string_view format_tier;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double dx_m = 0.0;
    double dy_m = 0.0;
    double dz_m = 0.0;
    double dt_large_s = 0.0;
    int acoustic_substeps = 0;
    double write_every_s = 0.0;
};

inline bool matches_qualified_radar_zfp_v4_context(
    const QualifiedRadarZfpV4Context& context)
{
    const bool cadence = context.write_every_s == 1 || context.write_every_s == 4;
    bool tier = false;
    for (const auto candidate : kQualifiedRadarZfpV4FormatTiers)
    {
        tier = tier || context.format_tier == candidate;
    }
    return context.qualified && context.cpu && context.fallback_off &&
        context.cartesian && context.c_grid && context.flat &&
        context.stationary && context.no_nesting && context.no_tracking &&
        context.no_side_stream && context.no_tiered_cadence &&
        context.no_live_shm && context.no_trajectories &&
        context.deterministic_reductions && context.reviewed_source_profile &&
        context.nx == 12 &&
        context.ny == 12 && context.nz == 48 &&
        context.dx_m == 1000.0 && context.dy_m == 1000.0 &&
        context.dz_m == 250.0 && context.dt_large_s == 0.2 &&
        context.acoustic_substeps == 6 && cadence && tier;
}

struct QualifiedRadarZfpV4CliOverrides
{
    bool dt_large_set = false;
    double dt_large_s = 0.0;
    bool acoustic_substeps_set = false;
    int acoustic_substeps = 0;
    bool write_every_set = false;
    int write_every_s = 0;
    bool live_shm_set = false;
    bool trajectory_set = false;
    bool guard_scope_weakened = false;
    bool threads_set = false;
    int threads = 0;
};

inline std::string_view qualified_radar_zfp_v4_cli_scope_error(
    const QualifiedRadarZfpV4CliOverrides& overrides)
{
    if (overrides.dt_large_set && overrides.dt_large_s != 0.2)
    {
        return "qualified radar ZFP v4 scope requires --dt-large=0.2";
    }
    if (overrides.acoustic_substeps_set && overrides.acoustic_substeps != 6)
    {
        return "qualified radar ZFP v4 scope requires --acoustic-substeps=6";
    }
    if (overrides.write_every_set && overrides.write_every_s != 1 &&
        overrides.write_every_s != 4)
    {
        return "qualified radar ZFP v4 scope requires --write-every=1 or 4";
    }
    if (overrides.live_shm_set)
    {
        return "qualified radar ZFP v4 scope does not admit live-SHM CLI overrides";
    }
    if (overrides.trajectory_set)
    {
        return "qualified radar ZFP v4 scope does not admit trajectory CLI overrides";
    }
    if (overrides.guard_scope_weakened)
    {
        return "qualified radar ZFP v4 scope requires --guard-scope=required";
    }
    if (overrides.threads_set && overrides.threads != 1 && overrides.threads != 4)
    {
        return "qualified radar ZFP v4 scope requires --threads=1 or 4";
    }
    return {};
}

} // namespace nimbus::output
