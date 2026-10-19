#pragma once

#include "core/runtime/generated_metal_output_scope_v2.hpp"
#include "diagnostics/validation_policy.hpp"

#include <string_view>

namespace tmv::runtime
{

struct QualifiedGeneratedMetalCliOverrides
{
    bool dt_large = false;
    double dt_large_s = 0.0;
    bool acoustic_substeps = false;
    int acoustic_substeps_value = 0;
    bool write_every = false;
    int write_every_s = 0;
    bool live_shm = false;
    bool trajectories = false;
    bool guard_mode = false;
    tmv::GuardMode guard_mode_value = tmv::GuardMode::Strict;
    bool guard_fail_on = false;
    tmv::GuardFailOn guard_fail_on_value = tmv::GuardFailOn::Both;
    bool guard_scope = false;
    tmv::StrictGuardScope guard_scope_value = tmv::StrictGuardScope::RequiredOnly;
};

inline std::string_view qualified_generated_metal_cli_scope_error(
    const QualifiedGeneratedMetalCliOverrides& cli,
    bool east_npy,
    bool mixed_npy,
    std::string_view output_format) noexcept
{
    if (cli.guard_mode && cli.guard_mode_value != tmv::GuardMode::Strict)
    {
        return "Qualified execution requires --guard-mode=strict; a CLI override cannot weaken the configured science guard.";
    }
    if (cli.guard_fail_on && cli.guard_fail_on_value != tmv::GuardFailOn::Both)
    {
        return "Qualified execution requires --guard-fail-on=both; a CLI override cannot weaken the configured science guard.";
    }
    if (cli.dt_large && cli.dt_large_s != 0.1)
    {
        return "qualified generated Metal scope requires --dt-large=0.1";
    }
    if (cli.acoustic_substeps && cli.acoustic_substeps_value != 6)
    {
        return "qualified generated Metal scope requires --acoustic-substeps=6";
    }
    if (mixed_npy && output_format != "npy_3d")
    {
        return "qualified mixed-open Metal scope requires synchronous minimal NPY";
    }
    if (mixed_npy && cli.write_every && cli.write_every_s != 1)
    {
        return "qualified mixed-open Metal NPY scope requires --write-every=1";
    }
    if (east_npy && cli.write_every && cli.write_every_s != 1)
    {
        return "qualified east-radiative Metal NPY scope requires --write-every=1";
    }
    if (cli.write_every && !admits_generated_metal_output_write_cadence(output_format, cli.write_every_s))
    {
        return "qualified generated Metal scope rejects an unreviewed --write-every cadence";
    }
    if (cli.live_shm)
    {
        return "qualified generated Metal scope does not admit live-SHM CLI overrides";
    }
    if (cli.trajectories)
    {
        return "qualified generated Metal scope does not admit trajectory CLI overrides";
    }
    if (cli.guard_scope && cli.guard_scope_value != tmv::StrictGuardScope::RequiredOnly)
    {
        return "qualified generated Metal scope requires --guard-scope=required";
    }
    return {};
}

}
