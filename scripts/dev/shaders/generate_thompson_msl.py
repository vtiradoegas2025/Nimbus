#!/usr/bin/env python3
"""Generate Metal from the canonical Thompson moment sedimentation GLSL."""

import argparse
from generate_tke_msl import ROOT, translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    names = ("thompson_moment_sedimentation", "thompson_cloud_adjustment", "thompson_warm_rain", "thompson_cloud_phase", "thompson_freezing", "thompson_deposition", "thompson_completion")
    parser.add_argument("--kernel", choices=names)
    args = parser.parse_args()
    for name in ((args.kernel,) if args.kernel else names):
        path = ROOT / f"metal/include/compute/metal/shaders/microphysics/{name}_msl.hpp"
        expected = translate(name, "microphysics")
        if args.check:
            if not path.is_file() or path.read_text() != expected:
                raise RuntimeError(f"generated Thompson Metal source drifted: {path}")
        else:
            path.write_text(expected)



if __name__ == "__main__":
    main()
