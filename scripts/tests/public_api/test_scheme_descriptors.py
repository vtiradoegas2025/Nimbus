#!/usr/bin/env python3
"""Compile and execute the public scheme-capability completeness gate."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path


SOURCE = r'''
#include "boundary_layer/scheme_descriptor.hpp"
#include "chaos/scheme_descriptor.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "microphysics/scheme_descriptor.hpp"
#include "numerics/advection/scheme_descriptor.hpp"
#include "numerics/diffusion/scheme_descriptor.hpp"
#include "numerics/time_stepping/scheme_descriptor.hpp"
#include "radar/scheme_descriptor.hpp"
#include "radiation/scheme_descriptor.hpp"
#include "soundings/scheme_descriptor.hpp"
#include "terrain/scheme_descriptor.hpp"
#include "turbulence/scheme_descriptor.hpp"

#include <iostream>
#include <set>
#include <string>
#include <vector>

int validate(const char* module,
             const std::vector<tmv::SchemeDescriptor>& descriptors)
{
    std::set<std::string> identities;
    for (const auto& descriptor : descriptors)
    {
        if (!tmv::has_complete_scheme_capabilities(descriptor))
        {
            std::cerr << module << ": incomplete capabilities for "
                      << descriptor.id << '\n';
            return 1;
        }
        if (!identities.insert(descriptor.id).second)
        {
            std::cerr << module << ": duplicate identity " << descriptor.id << '\n';
            return 1;
        }
        for (const auto& alias : descriptor.aliases)
        {
            if (!identities.insert(alias).second)
            {
                std::cerr << module << ": duplicate identity " << alias << '\n';
                return 1;
            }
        }
    }
    return 0;
}

int main()
{
    int result = 0;
    result |= validate("boundary_layer", tmv::get_boundary_layer_scheme_descriptors());
    result |= validate("chaos", tmv::chaos::get_chaos_scheme_descriptors());
    result |= validate("dynamics", tmv::get_dynamics_scheme_descriptors());
    result |= validate("microphysics", tmv::get_microphysics_scheme_descriptors());
    result |= validate("advection", tmv::get_advection_scheme_descriptors());
    result |= validate("diffusion", tmv::get_diffusion_scheme_descriptors());
    result |= validate("time_stepping", tmv::get_time_stepping_scheme_descriptors());
    result |= validate("radar", tmv::get_radar_scheme_descriptors());
    result |= validate("radiation", tmv::get_radiation_scheme_descriptors());
    result |= validate("soundings", tmv::get_sounding_scheme_descriptors());
    result |= validate("terrain", tmv::get_terrain_scheme_descriptors());
    result |= validate("turbulence", tmv::get_turbulence_scheme_descriptors());
    return result;
}
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    include_dir = root / "include"
    compiler = shlex.split(args.compiler)

    with tempfile.TemporaryDirectory(prefix="nimbus-scheme-descriptors-") as temp_dir:
        temp_path = Path(temp_dir)
        source = temp_path / "scheme_descriptors.cpp"
        binary = temp_path / "scheme_descriptors"
        source.write_text(SOURCE, encoding="utf-8")

        compile_result = subprocess.run(
            compiler
            + ["-std=c++17", "-I", str(include_dir), str(source), "-o", str(binary)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            print("[scheme-descriptors] FAIL: compile")
            print(compile_result.stdout, end="")
            print(compile_result.stderr, end="")
            return compile_result.returncode

        run_result = subprocess.run(binary, cwd=root, capture_output=True, text=True)
        if run_result.returncode != 0:
            print("[scheme-descriptors] FAIL: completeness")
            print(run_result.stdout, end="")
            print(run_result.stderr, end="")
            return run_result.returncode

    print("[scheme-descriptors] PASS: 12 public catalogs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
